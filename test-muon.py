from muon import Muon
import inspect, logging
print(inspect.signature(Muon.__init__))

avail = set(inspect.signature(Muon.__init__).parameters)
logging.info(f"Muon args available: {sorted(avail)}")
