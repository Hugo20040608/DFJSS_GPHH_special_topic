from GPHH import main
import time

if __name__ == '__main__':
    for i in range(1, 31):  #range次數可以改，看要重複跑幾次實驗
        start = time.time()
        main(run=i)
        end = time.time()
        print(f'Execution time simulation: {end - start}')    