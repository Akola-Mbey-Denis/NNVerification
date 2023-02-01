# Use this file instead of the example from the tutorial to cross check your results with the one
# in slide 24 of the class.

FORALL = "forall"
EXISTS = "exists"
p = 3
@variables x[1:p]
g = x[1]*x[1]/4.0 + (x[2] + 1)*(x[3] + 2) + (x[3] + 3) * (x[3] + 3)
quantifiers = [EXISTS, FORALL, EXISTS]