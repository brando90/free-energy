import sys; sys.dont_write_bytecode=True
import e10topup as T
led=T.ledger()
print(round(led.cost_usd()+led.extra_cost_usd(),4))
