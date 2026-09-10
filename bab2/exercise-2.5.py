import math

print('Enter float numbers separated by spaces: ')
x = input()
numbers = [float(num) for num in x.split()]

for num in numbers:
    sine_val = math.sin(num)
    print('The sine of ' + str(num) + ' is ' + str(sine_val))
