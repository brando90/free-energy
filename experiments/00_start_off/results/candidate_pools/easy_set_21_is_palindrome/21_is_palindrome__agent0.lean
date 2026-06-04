import Lean
import Mathlib.Tactic

/-
  Check if a non-negative integer is a palindrome.
  
  Edge cases:
  - Single digits are palindromes.
  - Negative inputs are invalid and raise ValueError.
-/

def getDigits (n : Nat) : List Nat :=
  if n < 10 then [n]
  else (getDigits (n / 10)) ++ [n % 10]

def is_palindrome (n : Int) : Except String Bool :=
  if n < 0 then
    Except.error "Input must be non-negative integer"
  else
    let digits := getDigits n.toNat
    Except.ok (digits == digits.reverse)

def pre (n : Int) : Prop :=
  n ≥ 0

def post (n : Int) (result : Bool) : Prop :=
  let digits := getDigits n.toNat
  result = (digits == digits.reverse)

theorem is_palindrome_correct (n : Int) (h : pre n) : 
  match is_palindrome n with
  | Except.ok result => post n result
  | Except.error _ => False
  := by sorry

-- Tests
#eval is_palindrome 121  -- Should return Except.ok true
#eval is_palindrome 0    -- Should return Except.ok true
#eval is_palindrome (-1) -- Should return Except.error "Input must be non-negative integer"

example : is_palindrome 121 = Except.ok true := by simp [is_palindrome]; rfl
example : is_palindrome 0 = Except.ok true := by simp [is_palindrome]; rfl
example : is_palindrome (-1) = Except.error "Input must be non-negative integer" := by simp [is_palindrome]; rfl
