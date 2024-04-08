#include <iostream>

// Function to perform Bubble Sort
void bubbleSort(int array[], int n) {
  
  for (int i = 0; i < n - 1; i++) {
    // Flag to check if any swap happens
    bool swapped = false;
        
    // Inner loop for comparing adjacent elements
    for (int j = 0; j < n - i - 1; j++) {
      if (array[j] > array[j + 1]) {
      // Swap if elements are in wrong order
      std::swap(array[j], array[j + 1]);
      swapped = true;
      }
    }

  // If no two elements were swapped, the array is sorted
  if (!swapped)
    break;
  
  }
}


// Function to perform Bubble Sort
void bubbleSortUdemy(int array[], int size) {
  
  for (int i = (size - 1); i > 0; i--) {
    // Flag to check if any swap happens
    bool swapped = false;   
    // Inner loop for comparing adjacent elements
    for (int j = 0; j < i; j++) {
      if (array[j] > array[j + 1]) {
        // Swap if elements are in wrong order
        std::swap(array[j], array[j + 1]);
        swapped = true;
      }
    }
    // If no two elements were swapped, the array is sorted
    if (!swapped)
      break;
    
    }
}


// Main function
int main() {
  
  int array[] = {64, 34, 25, 12, 22, 11, 90};
  int n = sizeof(array)/sizeof(array[0]);
  bubbleSort(array, n);
  bubbleSortUdemy(array, n);
  for(auto value: array){
    std::cout << value << " ";
  }
  std::cout << std::endl;

  return 0;

}





