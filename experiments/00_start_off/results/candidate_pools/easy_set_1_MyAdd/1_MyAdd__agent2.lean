import VeriBench

/-
Return the sum of two natural numbers (non-negative integers).

Edge cases:
- 0 + n = n
- The function raises ValueError if a or b is not a natural number.
-/

namespace sum_of_naturals

def pre (a : Int) (b : Int) : Prop :=
  a ≥ 0 ∧ b ≥ 0

def post (a : Int) (b : Int) (result : Int) : Prop :=
  result = a + b

def prog (a : Int) (b : Int) : Int :=
  if a < 0 ∨ b < 0 then
    panic! s!"Inputs must be non-negative integers (got a={a}, b={b})"
  else
    a + b

theorem prog_correct (a : Int) (b : Int) : pre a b → post a b (prog a b) := by
  intro h
  unfold pre at h
  unfold post
  unfold prog
  simp [h]
  sorry

-- Tests
def test : IO Unit := do
  -- Basic test
  VeriBench.assertEq (prog 1 2) 3
  
  -- Edge case
  VeriBench.assertEq (prog 0 0) 0

  -- Negative cases (pre-violations must raise errors)
  VeriBench.assertPanics (prog (-1) 0)
  VeriBench.assertPanics (prog 0 (-2))

  IO.println "All tests passed."

end sum_of_naturals
