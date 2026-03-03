import time, adafruit_dht, board
dht = adafruit_dht.DHT11(board.D4)
for _ in range(15):
    try:
        print("T:", dht.temperature, " H:", dht.humidity)
    except Exception as e:
        print("err:", e)
    time.sleep(1)