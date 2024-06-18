#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>
#include <csignal>
#include <optional>
#include <fstream>

using namespace std;

// Check Endianness

bool isLittleEndian(){

  // Both int and char[] share the same place in the memory thanks to union
  // So we assign the number into int but we can check its bytes with char[]
  union{
    uint32_t i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 4;

}

// or we can just use reinterpret_cast
bool isLittleEndian2() {
  uint32_t value = 0x01020304;
  char *bytePointer = reinterpret_cast<char*>(&value);
    
  // If the first byte is 0x04, it is little endian
  return bytePointer[0] == 0x04;
}


// Handle Endianness

uint32_t swapEndian(uint32_t val) {
  
  // Move byte 3 to byte 0
  uint32_t val1 = (val >> 24);      // 0x00'00'00'03
  val1 = val1 & 0x00'00'00'ff;      // 0x00'00'00'03 - Bit mask is not needed

  // Move byte 2 to byte 1
  uint32_t val2 = (val >> 8);       // 0x00'03'04'05
  val2 = val2 & 0x00'00'ff'00;      // 0x00'00'04'00

  // Move byte 1 to byte 2
  uint32_t val3 = (val << 8);       // 0x04'05'06'00
  val3 = val3 & 0x00'ff'00'00;      // 0x00'05'00'00

  // Move byte 0 to byte 3
  uint32_t val4 = (val << 24);      // 0x06'00'00'00
  val4 = val4 & 0xff'00'00'00;      // 0x06'00'00'00 - Bit mask is not needed

  val = val1 | val2 | val3 | val4;  // 0x06'05'04'03

  return val;
}



void writeLittleEndian(uint32_t value) {
  std::ofstream outFile("test.bin", std::ios::binary);
  if(outFile.is_open()){
    outFile.put(value & 0xFF);
    outFile.put((value >> 8) & 0xFF);
    outFile.put((value >> 16) & 0xFF);
    outFile.put((value >> 24) & 0xFF);
    outFile.close();
  }
}

uint32_t readLittleEndian() {
  uint32_t value = 0;
  std::ifstream inFile("test.bin", std::ios::binary);

  if(inFile.is_open()){
    value |= inFile.get();
    value |= inFile.get() << 8;
    value |= inFile.get() << 16;
    value |= inFile.get() << 24;
    inFile.close();
  }
  
  return value;
}

int main() {

  std::cout << isLittleEndian() << std::endl;
  uint32_t value = 0x03040506;



  return 0;
}




