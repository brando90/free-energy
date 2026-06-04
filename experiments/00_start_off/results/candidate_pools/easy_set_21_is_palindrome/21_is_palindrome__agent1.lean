import VeriBench

namespace IsPalindrome

/-- Check if a non-negative integer is a palindrome -/
def is_palindrome (n : Nat) : Bool :=
  if n < 10 then
    true
  else
    let digits := toString n
    digits == digits.reverse

/-- Precondition: Input must be a non-negative integer -/
def pre (n : Int) : Bool :=
  n ≥ 0 && n.isNat

/-- Postcondition: Returns true if n reads the same forward and backward, false otherwise -/
def post (n : Nat) (result : Bool) : Bool :=
  let digits := toString n
  result = (digits == digits.reverse)

/-- Correctness theorem: is_palindrome satisfies its postcondition -/
theorem is_palindrome_correct (n : Nat) :
  post n (is_palindrome n) := by
  sorry

/-- Tests for is_palindrome -/
def tests : VeriBench.Tests :=
  VeriBench.group "is_palindrome" [
    VeriBench.test "basic case" (is_palindrome 121 = true),
    VeriBench.test "edge case - single digit" (is_palindrome 0 = true),
    VeriBench.test "edge case - single digit" (is_palindrome 5 = true),
    VeriBench.test "non-palindrome" (is_palindrome 123 = false),
    VeriBench.test "large palindrome" (is_palindrome 12344321 = true),
    VeriBench.test "large non-palindrome" (is_palindrome 12345678 = false)
  ]

end IsPalindrome
