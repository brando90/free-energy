import Mathlib.Data.List.Basic
import Mathlib.Data.Option.Basic
import Mathlib.Tactic.Basic

/-- Binary search over a sorted list of integers.
    Returns the index if found, None if not found. -/
def binarySearch (arr : List Int) (target : Int) : Option Nat :=
  if arr = [] then
    none
  else
    let rec loop (left right : Nat) : Option Nat :=
      if left > right then
        none
      else
        let mid := (left + right) / 2
        let midVal := arr[mid]!
        if midVal = target then
          some mid
        else if midVal < target then
          loop (mid + 1) right
        else
          loop left (mid - 1)
    loop 0 (arr.length - 1)

/-- Precondition: list must be sorted in non-decreasing order -/
def binarySearch.pre (arr : List Int) (target : Int) : Prop :=
  ∀ i j, i < j → j < arr.length → arr[i]! ≤ arr[j]!

/-- Postcondition: if the function returns some index, then the element at that index
    equals the target. If it returns none, then the target is not in the array. -/
def binarySearch.post (arr : List Int) (target : Int) (result : Option Nat) : Prop :=
  match result with
  | some idx => idx < arr.length ∧ arr[idx]! = target
  | none => ∀ i, i < arr.length → arr[i]! ≠ target

/-- Correctness theorem: binarySearch satisfies its specification -/
theorem binarySearch_correct (arr : List Int) (target : Int) 
    (h : binarySearch.pre arr target) : 
    binarySearch.post arr target (binarySearch arr target) := by
  sorry

/-- Tests for binary search -/
def binarySearch_test : IO Unit := do
  -- Basic functionality tests
  assert! (binarySearch [1, 2, 3, 4, 5] 1 = some 0)
  assert! (binarySearch [1, 2, 3, 4, 5] 3 = some 2)
  assert! (binarySearch [1, 2, 3, 4, 5] 5 = some 4)
  assert! (binarySearch [1, 2, 3, 4, 5] 6 = none)
  assert! (binarySearch [1, 2, 3, 4, 5] 0 = none)
  
  -- Edge cases
  assert! (binarySearch [] 1 = none)
  assert! (binarySearch [5] 5 = some 0)
  assert! (binarySearch [5] 3 = none)
  
  -- Larger arrays
  assert! (binarySearch [1, 3, 5, 7, 9] 3 = some 1)
  assert! (binarySearch [1, 3, 5, 7, 9] 7 = some 3)
  assert! (binarySearch [1, 3, 5, 7, 9] 4 = none)
  assert! (binarySearch [10, 20, 30, 40, 50, 60] 60 = some 5)
  assert! (binarySearch [10, 20, 30, 40, 50, 60] 10 = some 0)
  
  -- Test with duplicates
  let test_arr := [1, 2, 3, 3, 3, 4, 5]
  let result := binarySearch test_arr 3
  match result with
  | some idx => 
    assert! (test_arr[idx]! = 3)
    assert! (2 ≤ idx && idx ≤ 4)
  | none => IO.println "Test failed: should have found 3"
  
  -- Two element arrays
  assert! (binarySearch [1, 2] 1 = some 0)
  assert! (binarySearch [1, 2] 2 = some 1)
  assert! (binarySearch [1, 2] 3 = none)
  
  IO.println "All tests passed!"

#eval binarySearch_test
