#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <stdexcept>

class Timer{

public:
  void start();
  double read_us();

private:
  std::chrono::high_resolution_clock::time_point t;

};

#endif
