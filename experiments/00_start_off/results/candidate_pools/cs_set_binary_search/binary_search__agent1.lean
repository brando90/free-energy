import Mathlib.Data.List.Basic
import Mathlib.Data.Option.Basic

def isSorted (arr : List Int) : Bool :=
  match arr with
  | [] => true
  | x :: xs => List.forall₂ (· ≤ ·) (x :: xs) xs

def binarySearchPre (arr : List Int) (target : Int) : Bool :=
  isSorted arr

def binarySearch (arr : List Int) (target : Int) : Option Nat :=
  if not (binarySearchPre arr target) then
    panic "Require sorted List[Int] and Int target"
  else if arr.isEmpty then
    none
  else
    let rec search (left right : Nat) : Option Nat :=
      if left > right then
        none
      else
        let mid := (left + right) / 2
        match arr.get? mid with
        | none => none
        | some midVal =>
          if midVal = target then
            some mid
          else if midVal < target then
            search (mid + 1) right
          else
            search left (mid - 1)
    
    search 0 (arr.length - 1)

/-- Specification: binary_search returns the index of target in the array if it exists, none otherwise -/
def binarySearchPost (arr : List Int) (target : Int) (result : Option Nat) : Prop :=
  match result with
  | none => ∀ i, i < arr.length → arr[i] ≠ target
  | some idx => idx < arr.length ∧ arr[idx] = target

theorem binarySearchCorrect (arr : List Int) (target : Int) 
    (h : binarySearchPre arr target) : 
    binarySearchPost arr target (binarySearch arr target) := sorry

-- Tests
#eval binarySearch [1, 2, 3, 4, 5] 3 -- some 2
#eval binarySearch [1, 2, 3, 4, 5] 6 -- none
#eval binarySearch [] 1 -- none
#eval binarySearch [5] 5 -- some 0
#eval binarySearch [5] 3 -- none
#eval binarySearch [1, 3, 5, 7, 9] 7 -- some 3
#eval binarySearch [10, 20, 30, 40, 50, 60] 60 -- some 5
#eval binarySearch [10, 20, 30, 40, 50, 60] 10 -- some 0
#eval binarySearch [1, 2, 3, 3, 3, 4, 5] 3 -- some 2, 3, or 4
#eval binarySearch [1, 2] 1 -- some 0
#eval binarySearch [1, 2] 2 -- some 1
