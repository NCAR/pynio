class nioDict (dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        #print 'in nioDict getitem', self.path, repr(self.topdict)
        if key[0] == '/' and len(key) > 1:
            val = dict.__getitem__(self, key[1:])
        elif key.find('/') > 0 and not self.topdict is None:
            newkey = ''.join([self.path,'/',key])
            val = dict.__getitem__(self.topdict, newkey)
        else:
            val = dict.__getitem__(self, key)
#        print 'GET', key
        return val

    def __setitem__(self, key, val):
#        print 'SET', key, val
        dict.__setitem__(self, key, val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        #return '%s(%s)' % (type(self).__name__, dictrepr)
        return  dictrepr

    def update(self, *args, **kwargs):
        #print 'update', args, kwargs
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
   

