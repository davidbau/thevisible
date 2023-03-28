j = 409
while j > 1:
    if j % 3 == 0:
        j //= 3
    elif j % 3 == 2:
        j = (j + 1) // 3
    else:
        j = j*5 + 1
    print(j)
