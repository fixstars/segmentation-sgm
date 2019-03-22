/*
Copyright 2019 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <limits>

namespace sgm
{

constexpr int P_OO1 = 10;        // object <=> object transition cost
constexpr int P_OO2 = 2 * P_OO1; // object <=> object transition cost when violate object ordering
constexpr int P_OG1 = 10;        // object <=> ground transition cost
constexpr int P_OG2 = 2 * P_OG1; // object <=> ground transition cost
constexpr int P_INF = std::numeric_limits<short>::max();

constexpr int MAX_DISPARITY = 128;
constexpr int MAX_SEGMENT_WIDTH = 16;

} // namespace sgm

#endif // !__PARAMETERS_H__
