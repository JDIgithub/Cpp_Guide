#include <iostream>
#include <thread>



void insertionSort(int array[], int size){
  
  for(size_t i{1}; i < size; ++i){
    int temp = array[i];
    int j = i - 1;  // Compare to previous element  

    // Move elements that are grater than temp
    // to one position ahead of their current position
    while(j >= 0 && temp < array[j] ) {
      array[j + 1] = array[j];
      array[j] = temp;
      j--;
    }
  }
}

int main() {

  int array[] = {6,4,2,5,1,3};
  int size = sizeof(array)/sizeof(array[0]);
  insertionSort(array,size);
  for(auto value: array){
    std::cout << value << " ";
  }
  std::cout << std::endl;
  return 0;
}