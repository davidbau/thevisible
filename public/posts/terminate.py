def does_terminate(j):
    seen = set()
    while j > 1:
        if j in seen:
            return False
        seen.add(j)

        if j % 3 == 0:
            j //= 3
        elif j % 3 == 2:
            j = (j + 1) // 3
        else:
            j = j * 5 + 1
        print(j)

    return True

for j in range(410, 1000):
#for j in range(427, 1000):
    if does_terminate(j):
        print(f"The loop terminates for j = {j}")
        break
