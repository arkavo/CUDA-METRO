def n1(n, size):
    row = n // size
    col = n % size
    n1 = ((row+1)*size + col+0 + size*size)%(size*size)
    n2 = ((row-1)*size + col+1 + size*size)%(size*size)
    n3 = ((row-0)*size + col-1 + size*size)%(size*size)

    return n1, n2, n3

print(n1(9, 3))