/-
Implementation of `sum_nat` function that adds two natural numbers.

Pre-condition: Both inputs must be natural numbers (non-negative integers).
Post-condition: The result is the sum of the inputs.
-/

import Mathlib.Data.Nat.Basic

def pre (a b : Int) : Prop :=
  a ≥ 0 ∧ b ≥ 0

def post (a b : Int) (result : Int) : Prop :=
  pre a b → result = a + b

def sum_nat (a b : Int) : Int :=
  if h : pre a b then
    a + b
  else
    panic! s!"Inputs must be non-negative integers (got a={a}, b={b})"

theorem sum_nat_correct (a b : Int) : post a b (sum_nat a b) := by
  unfold post
  intro h
  unfold sum_nat
  simp [h]
  sorry

/- Tests -/
#eval sum_nat 1 2 -- Should return 3
#eval sum_nat 0 0 -- Should return 0

example : sum_nat 1 2 = 3 := by rfl
example : sum_nat 0 0 = 0 := by rfl

/- Edge cases -/
example : sum_nat 0 5 = 5 := by rfl
example : sum_nat 5 0 = 5 := by rfl

/- Error handling for negative inputs is handled by the `panic!` in the implementation -/
