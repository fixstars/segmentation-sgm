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
