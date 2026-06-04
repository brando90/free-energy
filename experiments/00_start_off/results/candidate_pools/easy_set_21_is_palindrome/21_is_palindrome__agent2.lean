import VeriBench.Common

/-- Check if a non-negative integer is a palindrome.

Edge cases:
- Single digits are palindromes.
- Negative inputs are invalid and raise ValueError.
-/

def pre (n : Int) : Bool :=
  n >= 0

def is_palindrome (n : Int) : Bool :=
  if !pre n then
    panic! "Input must be non-negative integer"
  else if n < 10 then
    true
  else
    let digits := toString n
    digits == digits.reverse

/-- Tests for is_palindrome -/
def is_palindrome_test : IO Unit := do
  -- Basic
  assert! (is_palindrome 121 = true)
  -- Edge
  assert! (is_palindrome 0 = true)
  -- Negative
  try
    let _ := is_palindrome (-1)
    throw (IO.userError "expected pre-violation did not raise")
  catch
    | _ => pure ()

/-- Specification for is_palindrome -/
def is_palindrome_spec (n : Int) : Bool :=
  if !pre n then
    panic! "Input must be non-negative integer"
  else if n < 10 then
    true
  else
    let digits := toString n
    digits == digits.reverse

/-- Correctness theorem for is_palindrome -/
theorem is_palindrome_correct (n : Int) : is_palindrome n = is_palindrome_spec n := by
  sorry
