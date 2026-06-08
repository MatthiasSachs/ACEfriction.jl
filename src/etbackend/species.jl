# Species conversions for the ET backend (AtomsBase-based, no ACEfrictionCore).
# Species are represented internally as plain `Int` atomic numbers (the AtomsBase
# / ACEpotentials convention). These helpers convert between `Int` atomic numbers
# and element `Symbol`s (e.g. :Cu) for basis labels and user-facing input.
#
# Ported verbatim from ACEfrictionCore/src/utils/species.jl so the ET backend
# carries no dependency on ACEfrictionCore.

import AtomsBase
using AtomsBase: ChemicalSpecies

# Int atomic number -> element Symbol (e.g. 29 -> :Cu)
_chemical_symbol(z::Integer) = Symbol(ChemicalSpecies(z))
_chemical_symbol(s::Symbol) = s
_chemical_symbol(s::ChemicalSpecies) = Symbol(s)

# element Symbol / ChemicalSpecies -> Int atomic number (e.g. :Cu -> 29)
_atomic_number(z::Integer) = Int(z)
_atomic_number(s::Symbol) = Int(AtomsBase.atomic_number(s))
_atomic_number(s::ChemicalSpecies) = Int(AtomsBase.atomic_number(s))
