"""Frozen LOL-BO SELFIES VAE: SMILES <-> 256-d latent codes.

Wraps the vendored `InfoTransformerVAE` (bottleneck_size=2 x d_model=128 = 256-d
latent, GuacaMol-pretrained) with the exact encode/decode path LOL-BO uses in
`vendor/lolbo/molecule_objective.py`:

  encode  MoleculeObjective.vae_forward:  smiles -> sf.encoder -> split_selfies
          -> vocab ids -> collate_fn -> vae(tokens)['z'] -> reshape(-1, 256)
          (z is a *sample* from the posterior, not the mean -- upstream behaviour)
  decode  MoleculeObjective.vae_decode:   z.reshape(-1, 2, 128) -> vae.sample()
          -> dataobj.decode -> sf.decoder -> SMILES
          (vae.sample draws tokens with a hard Gumbel-softmax, so decoding is
          stochastic: the same z can decode to different molecules)

Differences from upstream, all forced and all inference-only:
  * `.cuda()` is replaced by an explicit device, so this runs on CPU for smoke
    tests and on GPU for real runs.
  * parameters are frozen (`requires_grad_(False)`) and the module stays in eval
    mode -- this package never fine-tunes the VAE.
  * molecules whose SELFIES tokens fall outside the checkpoint's fixed 97-token
    vocabulary are reported instead of raising KeyError (see verify_lolbo.py).
  * stereochemistry is stripped before encoding (see below).

Stereochemistry
---------------
The checkpoint's vocabulary has no stereo or bond-direction tokens at all
(`[C@H1]`, `[C@@H1]`, `[/C]`, `[\\C]`, `[S@]`, ...): LOL-BO trained the VAE on
GuacaMol SMILES without them. TRACE canonicalises with `isomericSmiles=True`, so
**59% of this benchmark's initial molecules are unencodable as written** and
upstream's `SELFIESDataset.encode` would raise KeyError on them. Passing the
molecule through `Chem.RemoveStereochemistry` first takes encodability to 100%
on both the initial data and a 2,000-molecule ZINC250k sample.

So `strip_stereo=True` is the default. This is not a handicap we impose: the
decoder cannot emit a stereocentre either, so the whole arm searches the flat
(stereo-free) subspace by construction. The oracle still scores the exact SMILES
the decoder produces, and each run records how many inputs needed stripping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem

import pl_shim

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
DEFAULT_STATE_DICT = VENDOR_DIR / "checkpoints" / "SELFIES-VAE-state-dict.pt"

# SELFIES VAE latent space: bottleneck_size (2) * d_model (128).
LATENT_DIM = 256
BOTTLENECK_SIZE = 2
D_MODEL = 128


def _import_vendor():
    pl_shim.install()
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from lolbo.utils.mol_utils.selfies_vae.data import SELFIESDataset, collate_fn
    from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import InfoTransformerVAE

    return SELFIESDataset, collate_fn, InfoTransformerVAE


def flatten_smiles(smiles: str) -> Optional[str]:
    """Stereo-free, non-isomeric SMILES, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


class EncodeResult:
    """Latent codes plus an account of which inputs could be encoded."""

    def __init__(
        self,
        z: torch.Tensor,
        encoded_smiles: List[str],
        failed: List[Tuple[str, str]],
        n_stereo_stripped: int = 0,
    ):
        self.z = z
        self.encoded_smiles = encoded_smiles
        self.failed = failed  # (smiles, reason)
        self.n_stereo_stripped = n_stereo_stripped

    @property
    def n_failed(self) -> int:
        return len(self.failed)


class FrozenSelfiesVAE:
    def __init__(
        self,
        state_dict_path: Path = DEFAULT_STATE_DICT,
        device: str = "cuda",
        max_string_length: int = 1024,
        strip_stereo: bool = True,
    ):
        import selfies as sf

        self.sf = sf
        SELFIESDataset, collate_fn, InfoTransformerVAE = _import_vendor()

        self.state_dict_path = Path(state_dict_path)
        if not self.state_dict_path.exists():
            raise FileNotFoundError(
                f"Missing SELFIES VAE checkpoint: {self.state_dict_path}. "
                "It ships in vendor/checkpoints/; see PROVENANCE.md."
            )

        self.device = torch.device(device)
        self.collate_fn = collate_fn
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)

        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        self.vae.load_state_dict(state_dict, strict=True)

        self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        # upstream sets this after construction (model default is 256)
        self.vae.max_string_length = int(max_string_length)

        self.dim = LATENT_DIM
        self.strip_stereo = bool(strip_stereo)
        self.n_stereo_stripped = 0
        self.smiles_to_selfies: Dict[str, str] = {}
        self.vocab = set(self.dataobj.vocab)

    # ------------------------------------------------------------------ encode

    def _tokens_for(self, smiles: str) -> Optional[torch.Tensor]:
        """SMILES -> vocab-id tensor, or None with the reason recorded by caller."""
        selfie = self.smiles_to_selfies.get(smiles)
        if selfie is None:
            encoder_input = smiles
            if self.strip_stereo:
                flat = flatten_smiles(smiles)
                if flat is None:
                    raise ValueError(f"RDKit could not parse SMILES: {smiles}")
                if flat != smiles:
                    self.n_stereo_stripped += 1
                encoder_input = flat
            selfie = self.sf.encoder(encoder_input)
            self.smiles_to_selfies[smiles] = selfie
        tokenized = self.dataobj.tokenize_selfies([selfie])[0]
        if any(token not in self.vocab for token in tokenized):
            return None
        return self.dataobj.encode(tokenized).unsqueeze(0)

    @torch.no_grad()
    def encode(self, smiles_list: Sequence[str], batch_size: int = 64) -> EncodeResult:
        """Posterior sample z for each encodable SMILES (LOL-BO vae_forward)."""
        pending: List[Tuple[str, torch.Tensor]] = []
        failed: List[Tuple[str, str]] = []
        stripped_before = self.n_stereo_stripped
        for smiles in smiles_list:
            try:
                tokens = self._tokens_for(smiles)
            except Exception as exc:  # selfies encoder failure
                failed.append((smiles, f"selfies_encoder:{type(exc).__name__}"))
                continue
            if tokens is None:
                failed.append((smiles, "out_of_vocab_token"))
                continue
            pending.append((smiles, tokens))

        encoded_smiles: List[str] = []
        chunks: List[torch.Tensor] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            X = self.collate_fn([tokens for _, tokens in batch]).to(self.device)
            out = self.vae(X)
            chunks.append(out["z"].reshape(-1, self.dim).detach().cpu())
            encoded_smiles.extend(smiles for smiles, _ in batch)

        z = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, self.dim)
        return EncodeResult(
            z=z.float(),
            encoded_smiles=encoded_smiles,
            failed=failed,
            n_stereo_stripped=self.n_stereo_stripped - stripped_before,
        )

    @torch.no_grad()
    def reconstruction_stats(self, smiles_list: Sequence[str], batch_size: int = 64) -> Dict[str, float]:
        """Teacher-forced reconstruction accuracy, for the install gate."""
        token_acc, string_acc, n_batches, n_encoded = 0.0, 0.0, 0, 0
        pending: List[torch.Tensor] = []
        for smiles in smiles_list:
            try:
                tokens = self._tokens_for(smiles)
            except Exception:
                continue
            if tokens is not None:
                pending.append(tokens)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            X = self.collate_fn(batch).to(self.device)
            out = self.vae(X)
            token_acc += float(out["recon_token_acc"])
            string_acc += float(out["recon_string_acc"])
            n_batches += 1
            n_encoded += len(batch)
        if n_batches == 0:
            return {"n_encoded": 0, "recon_token_acc": float("nan"), "recon_string_acc": float("nan")}
        return {
            "n_encoded": n_encoded,
            "recon_token_acc": token_acc / n_batches,
            "recon_string_acc": string_acc / n_batches,
        }

    # ------------------------------------------------------------------ decode

    @torch.no_grad()
    def decode(self, z: torch.Tensor, batch_size: int = 128) -> List[str]:
        """z -> SMILES (LOL-BO vae_decode). Undecodable entries come back as ''."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device).float()

        smiles_out: List[str] = []
        for start in range(0, z.shape[0], batch_size):
            chunk = z[start : start + batch_size]
            sample = self.vae.sample(z=chunk.reshape(-1, BOTTLENECK_SIZE, D_MODEL))
            for i in range(sample.size(-2)):
                try:
                    selfie = self.dataobj.decode(sample[i])
                    smiles = self.sf.decoder(selfie)
                except Exception:
                    smiles_out.append("")
                    continue
                if smiles:
                    self.smiles_to_selfies[smiles] = selfie
                smiles_out.append(smiles or "")
        return smiles_out

    @torch.no_grad()
    def sample_prior_z(self, n: int) -> torch.Tensor:
        return self.vae.sample_prior(n).reshape(-1, self.dim).detach().cpu().float()
