# test_max.py
from smbus2 import SMBus
BUS=1
for a in (0x57,0x5A):
    try:
        print("addr",hex(a),"part id:",hex(SMBus(BUS).read_byte_data(a,0xFF)))
    except Exception as e:
        print("addr",hex(a),"no reply:",e)
