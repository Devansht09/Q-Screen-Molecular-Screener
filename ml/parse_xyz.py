def parse_xyz_file(filepath: str):
    """
    Parse a QM9 XYZ file.

    Line 2 has many properties; we heuristically pick the first
    large negative value (<= -50) as the Hartree energy (u0).
    Second-to-last line contains SMILES as the first token.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 4:
            return None

        header_tokens = lines[1].strip().split()

        # Find first "big negative" number as u0 (in Hartree)
        nums = []
        for tok in header_tokens:
            try:
                nums.append(float(tok))
            except ValueError:
                continue

        u0_hartree = None
        for v in nums:
            if v <= -50:  # QM9 energies are large negative numbers
                u0_hartree = v
                break

        if u0_hartree is None:
            return None

        u0_kcal = u0_hartree * HARTREE_TO_KCAL

        # SMILES is on second-to-last line, first field
        smiles_line = lines[-2].strip()
        if not smiles_line:
            return None

        smiles = smiles_line.split()[0].strip()
        if not smiles:
            return None

        return smiles, u0_kcal

    except Exception:
        return None