for j in [409]:
    print('Starting j', j)
    count = 0
    seen = set()
    while j > 1 and j not in seen:
        seen.add(j)
        if j % 3 == 0:
            j //= 3
        elif j % 3 == 2:
            j = (j + 1) // 3
        else:
            j = j*5 + 1
        print(count, j)
        count += 1
        if count > 100:
            break
    if len(seen) > 30:
        print('got a long one.')
        break
