import VeriBench

namespace NaturalSum

/-- True iff both inputs are natural numbers. -/
def pre (a b : Int) : Bool :=
  a ≥ 0 && b ≥ 0

/-- Return the sum of two natural numbers. -/
def prog (a b : Int) : VeriBench.Result Int :=
  if pre a b then
    .ok (a + b)
  else
    .err s!"Inputs must be non-negative integers (got a={a}, b={b})"

/-- The postcondition: result is the sum of the inputs. -/
def post (a b : Int) (result : Int) : Prop :=
  result = a + b

/-- The correctness theorem: if preconditions are met, the program returns a result
    that satisfies the postcondition. -/
theorem prog_correct (a b : Int) :
  pre a b → match prog a b with
  | .ok result => post a b result
  | .err _ => False
  := by sorry

/-- Test cases for the natural sum function -/
def tests : List (VeriBench.TestCase (Int × Int) Int) := [
  -- Basic test
  ⟨(1, 2), .ok 3⟩,
  -- Edge test
  ⟨(0, 0), .ok 0⟩,
  -- Error cases
  ⟨(-1, 0), .err "Inputs must be non-negative integers (got a=-1, b=0)"⟩,
  ⟨(0, -2), .err "Inputs must be non-negative integers (got a=0, b=-2)"⟩
]

/-- Run the tests -/
def runTests : IO Unit :=
  VeriBench.checkTests tests (fun (a, b) => prog a b)

end NaturalSum
