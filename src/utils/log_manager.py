import os


def register(filename, typ, name):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    catalog = os.path.join(folder, 'catalog')
    basename = os.path.basename(filename)
    if not os.path.exists(catalog):
        with open(catalog, 'w') as f:
            f.write('filename,type,name\n')
            f.write('{},{},{}\n'.format(basename, typ, name))
    else:
        with open(catalog, 'a') as f:
            f.write('{},{},{}\n'.format(basename, typ, name))
