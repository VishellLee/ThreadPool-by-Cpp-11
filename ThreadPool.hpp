#pragma once
#include <thread>
#include <condition_variable>
#include <mutex>
#include <future>
#include <deque>
#include <vector>
#include <memory>
#include <functional>
#include <type_traits>

//线程池
class ThreadPool
{
public:
	using TaskType = std::function<void(void)>;

	ThreadPool(const std::size_t theadSize = std::thread::hardware_concurrency() - 1);

	~ThreadPool();

	//线程池提交任务函数，优先以值拷贝形式传参，若参数无拷贝构造函数则自动以引用形式传参，若传参为引用需确保多线程参数的生命周期
	//参数可用 std::ref 或 std::cref 包装以强制传递引用
	//用于 普通函数、函数对象、lambda表达式、类的静态成员函数
	template <typename F, typename... Args>
	auto submit(F&& f, Args&&... args)->std::future<decltype(f(std::forward<Args&>(args)...))>;

	//线程池提交任务函数，优先以值拷贝形式传参，若参数无拷贝构造函数则自动以引用形式传参，若传参为引用需确保多线程参数的生命周期
	//参数可用 std::ref 或 std::cref 包装以强制传递引用
	//用于 类的非静态成员函数
	template<typename F, typename ObjPtr, typename... Args>
	auto submit(F&& f, ObjPtr&& ptr, Args&&... args)->std::future<decltype((ptr->*f)(std::forward<Args&>(args)...))>;

	//线程池批量提交任务函数,tuple中的对象优先以值拷贝形式传参，若参数无拷贝构造函数则自动以引用形式传参，若传参为引用需确保多线程参数的生命周期
	//tuple中的对象可用 std::ref 或 std::cref 包装以强制传递引用
	//使用于任何可调用对象(普通函数、函数对象、lambda表达式、类的静态成员函数、类的非静态成员函数)
	template <typename... Args, typename ReturnType>
	void submitInBatch(std::vector<std::tuple<Args...>>& functors, std::vector<std::future<ReturnType>>& taskFutures);

	//关闭线程池
	void shutDown();

	std::size_t threadSize(void) const { return m_threadSize; }

private:
	//线程池调度函数
	void scheduled();

	//仿函数参数包装器，使用 SFINAE 对有拷贝构造的左值进行值传递
	template <typename T>
	static T& functorParameterWrapper(T& value,
		typename std::enable_if <std::is_copy_constructible<typename std::decay<T>::type>::value>::type* = nullptr)
	{
		return value;
	}

	//仿函数参数包装器，使用 SFINAE 对没有拷贝构造的左值, 若可以进行引用包装, 则包装为引用传递
	template <typename T>
	static std::reference_wrapper<T>
		functorParameterWrapper(T& value,
			typename std::enable_if<(std::is_object<typename std::decay<T>::type>::value || std::is_function<typename std::decay<T>::type>::value)
			&& !std::is_copy_constructible<typename std::decay<T>::type>::value>::type* = nullptr)
	{
		return std::ref(value);
	}

	//仿函数参数包装器，使用 SFINAE 对有拷贝构造的右值进行完美转发
	template <typename T>
	static T&& functorParameterWrapper(T&& value,
		typename std::enable_if<std::is_rvalue_reference<decltype(value)>::value &&
		(std::is_copy_constructible<typename std::decay<T>::type>::value ||
			std::is_move_constructible<typename std::decay<T>::type>::value)>::type* = nullptr)
	{
		return std::forward<T>(value);
	}

	//仿函数参数包装器，使用 SFINAE 对没有拷贝构造的右值, 若可以进行引用包装, 则包装为引用传递
	template <typename T>
	static std::reference_wrapper<T>
		functorParameterWrapper(T&& value,
			typename std::enable_if<std::is_rvalue_reference<decltype(value)>::value && (std::is_object<typename std::decay<T>::type>::value || std::is_function<typename std::decay<T>::type>::value)
			&& !std::is_copy_constructible<typename std::decay<T>::type>::value && !std::is_move_constructible<typename std::decay<T>::type>::value>::type* = nullptr)
	{
		return std::cref(value);
	}

	//通过类模板的偏特化进行tuple拆解与任务对象组装
	template <typename Tuple, typename ReturnType, bool Done, int Total, int... N>
	struct wrapperTasksImpl
	{
		static std::function<void(void)> wrapper(Tuple&& t, std::future<ReturnType>& tskFuture)
		{
			return wrapperTasksImpl<Tuple, ReturnType, Total == 2 + sizeof...(N), Total, N..., sizeof...(N)>::wrapper(std::forward<Tuple>(t), tskFuture);
		}
	};

	//通过类模板的偏特化进行tuple拆解与任务对象组装
	template <typename Tuple, typename ReturnType, int Total, int... N>
	struct wrapperTasksImpl<Tuple, ReturnType, true, Total, N...>
	{
		static std::function<void(void)> wrapper(Tuple&& t, std::future<ReturnType>& tskFuture)
		{
			auto tskFunc = std::bind(functorParameterWrapper(std::get<0>(std::forward<Tuple>(t))), functorParameterWrapper(std::get<N + 1>(std::forward<Tuple>(t)))...);
			std::shared_ptr<std::packaged_task<decltype(tskFunc())(void)>> tskPtr = std::make_shared<std::packaged_task<decltype(tskFunc())(void)>>(std::move(tskFunc));
			tskFuture = tskPtr->get_future();
			std::function<void(void)> task = [tskPtr]() {(*tskPtr)(); };

			return task;
		}
	};

	//将tuple内的 可调用对象 及 参数 包装为 线程池任务对象
	template <typename Tuple, typename ReturnType>
	std::function<void(void)> wrapperTasksFromTuple(Tuple&& t, std::future<ReturnType>& tskFuture)
	{
		using tuple_type = typename std::decay<Tuple>::type;
		return wrapperTasksImpl<Tuple, ReturnType, 1 == std::tuple_size<tuple_type>::value, std::tuple_size<tuple_type>::value>::wrapper(std::forward<Tuple>(t), tskFuture);
	}

	std::deque<TaskType> m_taskQue;
	std::vector<std::thread> m_theadVec;
	std::condition_variable cond_var;
	std::mutex m_mu;

	bool m_isShutDown;
	std::size_t m_threadSize;
};

inline ThreadPool::ThreadPool(const std::size_t theadSize)
	:m_isShutDown(false), m_threadSize(theadSize)
{
	if (m_threadSize < 0 || m_threadSize > 7)
	{
		m_threadSize = 3;
	}

	m_theadVec.reserve(m_threadSize);

	for (uint32_t i = 0; i < m_threadSize; i++)
	{
		m_theadVec.emplace_back(&ThreadPool::scheduled, this);
	}
}

inline ThreadPool::~ThreadPool()
{
	shutDown();
}

template<typename F, typename... Args> inline
auto ThreadPool::submit(F&& f, Args&& ...args) -> std::future<decltype(f(std::forward<Args&>(args)...))>
{
	using functor_result_type = typename std::result_of<F& (Args&...)>::type;

	std::function<functor_result_type(void)> funcTask = std::bind(functorParameterWrapper(std::forward<F>(f)), functorParameterWrapper(std::forward<Args>(args))...);
	std::shared_ptr<std::packaged_task<functor_result_type(void)>> taskPtr = std::make_shared<std::packaged_task<functor_result_type(void)>>(funcTask);
	TaskType wrapperTask = [taskPtr](void)->void { (*taskPtr)(); };

	{
		std::lock_guard<std::mutex> gurad(m_mu);
		m_taskQue.push_back(wrapperTask);
	}

	cond_var.notify_one();

	return taskPtr->get_future();
}

template<typename F, typename ObjPtr, typename... Args> inline
auto ThreadPool::submit(F&& f, ObjPtr&& ptr, Args&&... args) -> std::future<decltype((ptr->*f)(std::forward<Args&>(args)...))>
{
	using functor_result_type = decltype((ptr->*f)(std::forward<Args&>(args)...));

	std::function<functor_result_type(void)> funcTask = std::bind(functorParameterWrapper(std::forward<F>(f)), functorParameterWrapper(std::forward<ObjPtr>(ptr)), functorParameterWrapper(std::forward<Args>(args))...);
	std::shared_ptr<std::packaged_task<functor_result_type(void)>> taskPtr = std::make_shared<std::packaged_task<functor_result_type(void)>>(funcTask);
	TaskType wrapperTask = [taskPtr](void)->void { (*taskPtr)(); };

	{
		std::lock_guard<std::mutex> gurad(m_mu);
		m_taskQue.push_back(wrapperTask);
	}

	cond_var.notify_one();

	return taskPtr->get_future();
}

template<typename... Args, typename ReturnType> inline
void ThreadPool::submitInBatch(std::vector<std::tuple<Args...>>& functors, std::vector<std::future<ReturnType>>& taskFutures)
{
	std::deque<std::function<void(void)>> tasks;

	for (auto& functor : functors)
	{
		std::future<ReturnType> taskFuture;
		tasks.emplace_back(wrapperTasksFromTuple(functor, taskFuture));
		taskFutures.emplace_back(std::move(taskFuture));
	}

	{
		std::lock_guard<std::mutex> guard(m_mu);
		m_taskQue.insert(m_taskQue.end(), tasks.begin(), tasks.end());
	}

	if (tasks.size() == 1)
	{
		cond_var.notify_one();
	}
	else if (tasks.size() > 1)
	{
		cond_var.notify_all();
	}
}

inline void ThreadPool::scheduled()
{
	while (true)
	{
		std::unique_lock<std::mutex> guard(m_mu);

		cond_var.wait(guard, [this]() {return !m_taskQue.empty() || m_isShutDown; });
		if (m_isShutDown)
		{
			break;
		}

		auto task = m_taskQue.front();
		m_taskQue.pop_front();
		guard.unlock();
		task();
	}
}

inline void ThreadPool::shutDown()
{
	{
		std::lock_guard<std::mutex> gurad(m_mu);
		m_isShutDown = true;
	}

	cond_var.notify_all();

	for (std::size_t i = 0; i < m_threadSize; i++)
	{
		if (m_theadVec[i].joinable())
		{
			m_theadVec[i].join();
		}
	}
}
