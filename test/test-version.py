import Nio, os, sys
from distutils.sysconfig import get_python_lib

print("Nio version = '"+Nio.__version__+"'")
print("Nio path = '"+Nio.__file__+"'")

os.system("ls -l " + Nio.__file__)

plib = get_python_lib()


if sys.platform == "darwin":
  os.system("otool -L " + plib + "/PyNIO/" + "_nio*.so")
else:
  os.system("ldd " + plib + "/PyNIO/" + "_nio.so")
