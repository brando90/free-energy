import Lean
import Mathlib.Data.List.Basic

def isSorted (l : List Int) : Bool :=
  match l with
  | [] => true
  | [_] => true
  | x :: y :: xs => x ≤ y && isSorted (y :: xs)

def pre (arr : List Int) (target : Int) : Bool := isSorted arr

def binarySearch (arr : List Int) (target : Int) : Option Nat :=
  if not (pre arr target) then
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

-- Tests
def test1 : binarySearch [1, 2, 3, 4, 5] 1 = some 0 := by rfl
def test2 : binarySearch [1, 2, 3, 4, 5] 3 = some 2 := by rfl
def test3 : binarySearch [1, 2, 3, 4, 5] 5 = some 4 := by rfl
def test4 : binarySearch [1, 2, 3, 4, 5] 6 = none := by rfl
def test5 : binarySearch [1, 2, 3, 4, 5] 0 = none := by rfl
def test6 : binarySearch ([] : List Int) 1 = none := by rfl
def test7 : binarySearch [5] 5 = some 0 := by rfl
def test8 : binarySearch [5] 3 = none := by rfl
def test9 : binarySearch [1, 3, 5, 7, 9] 3 = some 1 := by rfl
def test10 : binarySearch [10, 20, 30, 40, 50, 60] 60 = some 5 := by rfl
def test11 : binarySearch [1, 2] 1 = some 0 := by rfl
def test12 : binarySearch [1, 2] 2 = some 1 := by rfl
def test13 : binarySearch [1, 2] 3 = none := by rfl

-- Verification
theorem binarySearch_none_of_empty (target : Int) : 
  binarySearch [] target = none := by rfl

theorem binarySearch_some_of_found (arr : List Int) (target : Int) (idx : Nat) :
  pre arr target →
  idx < arr.length →
  arr.get idx = target →
  (∀ j, j < idx → arr.get j < target) →
  binarySearch arr target = some idx := sorry

theorem binarySearch_none_of_not_found (arr : List Int) (target : Int) :
  pre arr target →
  (∀ i, i < arr.length → arr.get i ≠ target) →
  binarySearch arr target = none := sorry

theorem binarySearch_correctness (arr : List Int) (target : Int) :
  pre arr target →
  match binarySearch arr target with
  | none => ∀ i, i < arr.length → arr.get i ≠ target
  | some idx => idx < arr.length ∧ arr.get idx = target
  := sorry
