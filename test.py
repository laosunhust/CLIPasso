import importlib.util, diffvg, pydiffvg
s = importlib.util.find_spec("diffvg")
print("origin:", getattr(s,"origin",None))
print("FilterType:", hasattr(diffvg,"FilterType"))
print("pydiffvg:", pydiffvg.__file__)