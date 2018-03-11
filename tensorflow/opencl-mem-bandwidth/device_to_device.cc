#include "clMemTester.h"
#include <vector>

int main(){

  std::vector<unsigned long int> Mbytes;

  for( auto i = 4 ; i < MAX_ALLOCATED ; i=i+10 ){
    Mbytes.push_back(i);
  }

  clMemTester c = clMemTester(NUM_OF_TESTS);

  c.init();

  for( auto b : Mbytes){
    c.DeviceToDevice( b << 20 );
  }

  c.clEnd();

  return 0;
}
