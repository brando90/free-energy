# TLDR: Idempotently patch an alexiglad/EBT checkout to register the 'veribench' dataset.
"""Apply the VeriBench integration to an official EBT checkout (tested at commit 19420cb).

Usage: python apply_ebt_integration.py /path/to/EBT
Copies veribench_dataloader.py into EBT/data/nlp/ and inserts the 'veribench'
branches into base_model_trainer.py's fit/test dataset dispatch.
"""
import shutil
import sys
from pathlib import Path

IMPORT_ANCHOR = "from data.nlp.fineweb_dataloader import FineWebDataset"
IMPORT_ADD = "from data.nlp.veribench_dataloader import VeribenchDataset"

FIT_ANCHOR = '''            else:
                raise NotImplementedError("Haven't implemented this dataset yet")'''
FIT_ADD = '''            elif self.hparams.dataset_name == "veribench":
                self.train_ds = VeribenchDataset(self.hparams, split="train")
                self.val_ds = VeribenchDataset(self.hparams, split="val")
'''

TEST_ANCHOR = '''            else:
                raise NotImplementedError("haven't implemented this dataset yet")'''
TEST_ADD = '''            elif self.hparams.dataset_name == "veribench":
                self.test_ds = VeribenchDataset(self.hparams, split="test")
'''


def main() -> int:
    ebt_root = Path(sys.argv[1] if len(sys.argv) > 1 else "EBT").resolve()
    trainer = ebt_root / "base_model_trainer.py"
    src = trainer.read_text()
    shutil.copy(Path(__file__).parent / "veribench_dataloader.py", ebt_root / "data" / "nlp" / "veribench_dataloader.py")
    if "VeribenchDataset" in src:
        print(f"[apply] {trainer} already patched — refreshed dataloader copy only")
        return 0
    for anchor, label in ((IMPORT_ANCHOR, "import"), (FIT_ANCHOR, "fit dispatch"), (TEST_ANCHOR, "test dispatch")):
        if src.count(anchor) != 1:
            raise SystemExit(f"[apply] {label} anchor not found exactly once (count={src.count(anchor)}) — EBT version drift?")
    src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + IMPORT_ADD)
    src = src.replace(FIT_ANCHOR, FIT_ADD + FIT_ANCHOR)
    src = src.replace(TEST_ANCHOR, TEST_ADD + TEST_ANCHOR)
    trainer.write_text(src)
    print(f"[apply] patched {trainer} (import + fit/test dispatch) and installed data/nlp/veribench_dataloader.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
