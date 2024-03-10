
#include <thread>
#include <iostream>

int search(int arr[],int n,int x){
  int i;
  for(int i = 0; i < n; i++){
    if(arr[i] == x){
      return i;
    }
  }
  return -1;
}

int main()
{
  int arr[] = { 2, 3, 4, 10, 40 };
  int n = sizeof(arr)/sizeof(arr[0]);
  int x = 10;
  int result = search(arr,n,x);

  if(result != -1){
    std::cout << "Element " << x <<  " is present at index " << result << std::endl;
  } else {
    std::cout << "Element " << x <<  " is not present" << std::endl;
  }
}