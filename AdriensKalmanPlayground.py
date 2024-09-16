from AdriensFunctions import *

# Low pass filter 
def LPF(data, alpha, initial = 'NA'):
    if initial == 'NA':
        start = data[0]
    else:
        start = initial
    filtered_data = [start]
    for k, datapoint in enumerate(data[1:]):
        filtered_data.append(alpha*filtered_data[k] + (1-alpha)*datapoint) 
    print(filtered_data)

def main():
    LPF([1,2,3,4,5,6,7,8,9],0.9, 5) 

if __name__ == "__main__":
    main()
