#include <chrono>
#include "Timer.h"

void Timer::start(){
  t = std::chrono::high_resolution_clock::now();
}

double Timer::read_us(){
  auto elapsed_time = std::chrono::high_resolution_clock::now() - t;
  auto time_us = std::chrono::duration<double,std::micro>(elapsed_time).count();
  return time_us;
}
