#include <iostream>

consteval int get_value(){
    
    int x(2.9);
    return x;
}

int main(){
    constexpr int value = get_value();
    std::cout << "value : " << value << std::endl;
    return 0;
}