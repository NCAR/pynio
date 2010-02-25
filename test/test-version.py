import Nio, os

print "Nio version = '"+Nio.__version__+"'"
print "Nio path = '"+Nio.__file__+"'"

os.system("ls -l " + Nio.__file__)
