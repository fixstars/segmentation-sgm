#ifndef __INTERVAL_TIMER_H__
#define __INTERVAL_TIMER_H__

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#define USE_STEADY_CLOCK
#ifdef USE_STEADY_CLOCK
using type_point = decltype(std::chrono::steady_clock::now());
static inline type_point now() { return std::chrono::steady_clock::now(); }
#else
using type_point = decltype(std::chrono::system_clock::now());
static inline type_point now() { return std::chrono::system_clock::now(); }
#endif

class IntervalTimer
{
public:

	static const int RESERVE_COUNT = 100;

	IntervalTimer()
	{
		reset();
		times_.reserve(RESERVE_COUNT);
		names_.reserve(RESERVE_COUNT);
	}

	void start(const char* name)
	{
		const type_point time = now();
		times_.push_back(time);
		names_.push_back(name);
	}

	void print() const
	{
		const type_point last = now();

		if (times_.empty())
			return;

		std::puts("=====================================");
		for (size_t i = 0; i < times_.size(); i++)
		{
			const type_point t0 = times_[i];
			const type_point t1 = i < times_.size() - 1 ? times_[i + 1] : last;
			const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
			std::printf("%25s : %.1f\n", names_[i].c_str(), 1e-3 * duration);
		}
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(last - times_[0]).count();
		std::printf("%25s : %.1f\n\n", "Total", 1e-3 * duration);
	}

	void reset()
	{
		times_.clear();
		names_.clear();
	}

private:
	std::vector<type_point> times_;
	std::vector<std::string> names_;
};

class CyclicTimer
{
public:

	static inline uint64_t duration_micro(const type_point& t0, const type_point& t1)
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	}

	static inline double duration_milli(const type_point& t0, const type_point& t1)
	{
		return 1e-3 * duration_micro(t0, t1);
	}

	static const int RESERVE_COUNT = 100;

	CyclicTimer(int cycle = 10, int waituntil = 0)
	{
		cycle_ = cycle;
		waituntil_ = waituntil;
		times_.reserve(RESERVE_COUNT);
		names_.reserve(RESERVE_COUNT);
		init();
	}

	void init()
	{
		count_ = 0;
		pos_ = 0;
		call_ = 0;
		active_ = false;
	}

	void add(const char* name, double duration)
	{
		if (times_.size() > pos_)
		{
			times_[pos_] += duration;
			tt_[pos_] += duration * duration;
			maxv_[pos_] = std::max(maxv_[pos_], duration);
		}
		else
		{
			times_.push_back(duration);
			tt_.push_back(duration * duration);
			maxv_.push_back(duration);
			names_.push_back(name);
		}
	}

	void start(const char* name)
	{
		if (!active_)
			return;

		const type_point t1 = now();
		add(name, pos_ > 0 ? duration_milli(t0_, t1) : 0);
		pos_++;
		t0_ = t1;
	}

	void update()
	{
		if (!active_)
		{
			active_ = ++call_ >= waituntil_;
			return;
		}

		const type_point t1 = now();
		add("", duration_milli(t0_, t1));
		pos_ = 0;
		if (++count_ % cycle_ == 0)
			print();
	}

	void print() const
	{
		if (times_.empty())
			return;

		std::printf("=============== Count %d ===============\n", count_);
		double sum = 0;
		for (size_t i = 0; i < times_.size() - 1; i++)
		{
			const double duration = times_[i + 1];
			const double duration_2 = tt_[i + 1];

			const double mean = duration / count_;
			const double variance = duration_2 / count_ - mean * mean;
			const double stddev = sqrt(variance);

			std::printf("%25s : %4.1f[msec], stddev(x2): %4.1f[msec] max: %4.1f[msec]\n",
				names_[i].c_str(), mean, 2 * stddev, maxv_[i + 1]);
			sum += duration;
		}
		std::printf("%25s : %4.1f[msec]\n\n", "Total", sum / count_);
	}

private:
	type_point t0_;
	std::vector<double> times_;
	std::vector<double> tt_;
	std::vector<double> maxv_;
	std::vector<std::string> names_;
	int cycle_, count_;
	int waituntil_, call_;
	size_t pos_;
	bool active_;
};

#endif // !__INTERVAL_TIMER_H__
