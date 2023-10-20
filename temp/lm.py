import math

def score(array_length, array):
    score = 0

    if array_length < 2:
        return score
    elif array_length < 3:
        for i in range(array_length-1):
            if (array[i] + array[i+1]) % 2 == 0:
                score += 5
        return score


    sum_of_couples= [0] * (array_length - 1)

    for i in range(array_length - 1):
        sum_of_couples[i] = array[i] + array[i+1]
    
    sum_of_triples = [0] * (array_length - 2)
    product_of_triples = [0] * (array_length - 2)
    for i in range(array_length - 2):
        sum_of_triples[i] = array[i] + array[i+1] + array[i+2]
        product_of_triples[i] = array[i] * array[i+1] * array[i+2]
    
    for i in sum_of_couples:
        if i % 2 == 0:
            score += 5
    
    for i in range(len(sum_of_triples)):
        if sum_of_triples[i] % 2 != 0 and product_of_triples[i] % 2 == 0:
            score += 10
    
    return score

arr = [1, 2,3,4]
print(score(len(arr), arr))