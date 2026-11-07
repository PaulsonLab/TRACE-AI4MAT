"""
Graph-GA molecule operators adapted from the MIT-licensed PMO/mol_opt
implementation, which itself follows Jensen, Chem. Sci. 2019, 10, 3567.

Local changes:
- no joblib dependency;
- constraint-aware molecule validation;
- canonical SMILES helpers for the benchmark harness.
"""

import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .common import MoleculeConstraints, canonicalize_smiles, molecule_satisfies_constraints


RDLogger.DisableLog("rdApp.error")
MINIMUM = 1e-10


def ring_ok(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
    rings = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(ring) for ring in rings]) if rings else 0
    macro_cycle = max_cycle_length > 6
    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_ok(mol: Chem.Mol, constraints: MoleculeConstraints) -> bool:
    return molecule_satisfies_constraints(mol, constraints) and ring_ok(mol)


def cut(mol: Chem.Mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    atom_pair = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))
    bond_ids = [mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def cut_ring(mol: Chem.Mol):
    for _ in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            atoms = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
            atom_pairs = ((atoms[0], atoms[1]), (atoms[2], atoms[3]))
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            atoms = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
            atom_pairs = ((atoms[0], atoms[1]), (atoms[1], atoms[2]))

        bond_ids = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in atom_pairs]
        fragments_mol = Chem.FragmentOnBonds(
            mol,
            bond_ids,
            addDummies=True,
            dummyLabels=[(1, 1), (1, 1)],
        )

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None

    return None


def crossover_non_ring(parent_a: Chem.Mol, parent_b: Chem.Mol, constraints: MoleculeConstraints):
    for _ in range(10):
        fragments_a = cut(parent_a)
        fragments_b = cut(parent_b)
        if fragments_a is None or fragments_b is None:
            return None

        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mols = []
        for frag_a in fragments_a:
            for frag_b in fragments_b:
                products = rxn.RunReactants((frag_a, frag_b))
                if products:
                    mol = products[0][0]
                    if mol_ok(mol, constraints):
                        new_mols.append(mol)

        if new_mols:
            return random.choice(new_mols)

    return None


def crossover_ring(parent_a: Chem.Mol, parent_b: Chem.Mol, constraints: MoleculeConstraints):
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_a.HasSubstructMatch(ring_smarts) and not parent_b.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts_1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    rxn_smarts_2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]

    for _ in range(10):
        fragments_a = cut_ring(parent_a)
        fragments_b = cut_ring(parent_b)
        if fragments_a is None or fragments_b is None:
            return None

        trial_mols = []
        for smarts in rxn_smarts_1:
            rxn = AllChem.ReactionFromSmarts(smarts)
            for frag_a in fragments_a:
                for frag_b in fragments_b:
                    products = rxn.RunReactants((frag_a, frag_b))
                    if products:
                        trial_mols.append(products[0])

        new_mols = []
        for smarts in rxn_smarts_2:
            rxn = AllChem.ReactionFromSmarts(smarts)
            for product in trial_mols:
                mol = product[0]
                if mol_ok(mol, constraints):
                    new_mols.extend(rxn.RunReactants((mol,)))

        final_mols = []
        for product in new_mols:
            mol = product[0]
            if mol_ok(mol, constraints):
                final_mols.append(mol)

        if final_mols:
            return random.choice(final_mols)

    return None


def crossover(parent_a: Chem.Mol, parent_b: Chem.Mol, constraints: MoleculeConstraints):
    parent_smiles = {Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)}
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except ValueError:
        pass

    for _ in range(10):
        if random.random() <= 0.5:
            child = crossover_non_ring(parent_a, parent_b, constraints)
        else:
            child = crossover_ring(parent_a, parent_b, constraints)

        if child is not None:
            smiles = Chem.MolToSmiles(child)
            if smiles is not None and smiles not in parent_smiles:
                return child

    return None


def delete_atom_smarts():
    choices = [
        "[*:1]~[D1:2]>>[*:1]",
        "[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]",
        "[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]",
        "[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]",
        "[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]",
    ]
    probs = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    return np.random.choice(choices, p=probs)


def append_atom_smarts():
    choices = [
        ("single", ["C", "N", "O", "F", "S", "Cl", "Br"]),
        ("double", ["C", "N", "O"]),
        ("triple", ["C", "N"]),
    ]
    bond_order, atom_list = choices[np.random.choice([0, 1, 2], p=[0.60, 0.35, 0.05])]
    new_atom = np.random.choice(atom_list)

    if bond_order == "single":
        return "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if bond_order == "double":
        return "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    return "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)


def insert_atom_smarts():
    choices = [
        ("single", ["C", "N", "O", "S"]),
        ("double", ["C", "N"]),
        ("triple", ["C"]),
    ]
    bond_order, atom_list = choices[np.random.choice([0, 1, 2], p=[0.60, 0.35, 0.05])]
    new_atom = np.random.choice(atom_list)

    if bond_order == "single":
        return "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if bond_order == "double":
        return "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    return "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)


def change_bond_order_smarts():
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    return np.random.choice(choices, p=[0.45, 0.45, 0.05, 0.05])


def add_ring_smarts():
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    return np.random.choice(choices, p=[0.05, 0.05, 0.45, 0.45])


def change_atom_smarts(mol: Chem.Mol):
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    probs = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    old_atom = np.random.choice(choices, p=probs)
    tries = 0
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + old_atom + "]")) and tries < 20:
        old_atom = np.random.choice(choices, p=probs)
        tries += 1

    new_atom = np.random.choice(choices, p=probs)
    while new_atom == old_atom:
        new_atom = np.random.choice(choices, p=probs)

    return "[X:1]>>[Y:1]".replace("X", old_atom).replace("Y", new_atom)


def mutate(mol: Chem.Mol, mutation_rate: float, constraints: MoleculeConstraints):
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    operator_probs = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(10):
        smarts_list = [
            insert_atom_smarts(),
            change_bond_order_smarts(),
            "[*:1]@[*:2]>>([*:1].[*:2])",
            add_ring_smarts(),
            delete_atom_smarts(),
            change_atom_smarts(mol),
            append_atom_smarts(),
        ]
        rxn_smarts = np.random.choice(smarts_list, p=operator_probs)
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        products = rxn.RunReactants((mol,))

        new_mols = []
        for product in products:
            trial = product[0]
            if mol_ok(trial, constraints):
                new_mols.append(trial)

        if new_mols:
            return random.choice(new_mols)

    return None


def make_mating_pool(population_smiles: List[str], population_scores: List[float], offspring_size: int):
    scores = np.asarray(population_scores, dtype=float)
    bottom = min(0.0, float(np.min(scores)))
    weights = scores - bottom + MINIMUM
    probs = weights / np.sum(weights)
    return list(np.random.choice(population_smiles, p=probs, size=offspring_size, replace=True))


def reproduce(mating_pool: List[str], mutation_rate: float, constraints: MoleculeConstraints) -> Optional[str]:
    parent_a = Chem.MolFromSmiles(random.choice(mating_pool))
    parent_b = Chem.MolFromSmiles(random.choice(mating_pool))
    if parent_a is None or parent_b is None:
        return None

    child = crossover(parent_a, parent_b, constraints)
    if child is not None:
        child = mutate(child, mutation_rate, constraints)
    if child is None or not mol_ok(child, constraints):
        return None

    return canonicalize_smiles(Chem.MolToSmiles(child, isomericSmiles=True))


def sanitize_population(smiles_list: Iterable[str], constraints: MoleculeConstraints) -> List[str]:
    out = []
    seen = set()
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is None or canonical in seen:
            continue
        mol = Chem.MolFromSmiles(canonical)
        if mol_ok(mol, constraints):
            seen.add(canonical)
            out.append(canonical)
    return out


def propose_graph_ga_offspring(
    population_smiles: List[str],
    population_scores: List[float],
    constraints: MoleculeConstraints,
    offspring_size: int,
    mutation_rate: float,
    max_attempt_multiplier: int = 30,
    exclude_smiles: Optional[Iterable[str]] = None,
) -> List[str]:
    if not population_smiles:
        return []

    offspring = []
    seen = set(population_smiles)
    for smiles in exclude_smiles or []:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            seen.add(canonical)
    attempts = 0
    max_attempts = max(offspring_size * max_attempt_multiplier, 100)

    while len(offspring) < offspring_size and attempts < max_attempts:
        attempts += 1
        mating_pool = make_mating_pool(population_smiles, population_scores, max(2, len(population_smiles)))
        child = reproduce(mating_pool, mutation_rate, constraints)
        if child is None or child in seen:
            continue
        seen.add(child)
        offspring.append(child)

    return offspring


def trim_population(
    candidate_smiles: List[str],
    score_lookup: Callable[[str], Optional[float]],
    population_size: int,
) -> Tuple[List[str], List[float]]:
    pairs = []
    seen = set()
    for smiles in candidate_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is None or canonical in seen:
            continue
        score = score_lookup(canonical)
        if score is None:
            continue
        seen.add(canonical)
        pairs.append((float(score), canonical))

    pairs.sort(reverse=True, key=lambda item: item[0])
    pairs = pairs[:population_size]
    return [smiles for _, smiles in pairs], [score for score, _ in pairs]
