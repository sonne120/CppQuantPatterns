
#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

template <typename S>
concept StochasticDifferentialEquation = requires(const S s, double x, double t)
{
    { s.Drift(x, t) }          -> std::convertible_to<double>;
    { s.Diffusion(x, t) }      -> std::convertible_to<double>;
    { s.InitialCondition() }   -> std::convertible_to<double>;
    { s.Expiry() }             -> std::convertible_to<double>;
};

template <typename R>
concept RandomNumberGenerator = requires(R r)
{
    { r() } -> std::convertible_to<double>;   
};

template <typename F, typename S>
concept FiniteDifferenceScheme = requires(const F f, double x, double t, double z)
{
    requires StochasticDifferentialEquation<S>;
    { f.advance(x, t, z) } -> std::convertible_to<double>;
    { f.steps() }          -> std::convertible_to<int>;
    { f.dt() }             -> std::convertible_to<double>;
};

template <typename P>
concept Pricer = requires(P p, const std::vector<double>& path)
{
    p.ProcessPath(path);
    p.PostProcess();
    { p.Price() } -> std::convertible_to<double>;
};

// Sde system
class GBM
{
    double mu{}, vol{}, d{}, ic{}, T{};
public:
    GBM(double drift, double diffusion, double dividend,
        double initialCondition, double expiry)
        : mu(drift), vol(diffusion), d(dividend),
          ic(initialCondition), T(expiry) {}

    double Drift(double x, double) const     { return (mu - d) * x; }
    double Diffusion(double x, double) const { return vol * x; }
    double InitialCondition() const          { return ic; }
    double Expiry() const                    { return T; }
};
static_assert(StochasticDifferentialEquation<GBM>);

// Fdm system
template <StochasticDifferentialEquation Sde>
class EulerFdm
{
    Sde sde;
    int NT;
    double k, dtSqrt;
public:
    EulerFdm(Sde s, int numSubdivisions)
        : sde(std::move(s)), NT(numSubdivisions),
          k(sde.Expiry() / NT), dtSqrt(std::sqrt(k)) {}

    double advance(double xn, double tn, double z) const
    {
        return xn + sde.Drift(xn, tn) * k
                  + sde.Diffusion(xn, tn) * dtSqrt * z;
    }
    int    steps() const { return NT; }
    double dt()    const { return k; }
    const Sde& equation() const { return sde; }
};
static_assert(FiniteDifferenceScheme<EulerFdm<GBM>, GBM>);

// Rng system
template <typename Engine = std::mt19937_64>
class NormalRng
{
    Engine eng;
    std::normal_distribution<double> nor{0.0, 1.0};
public:
    explicit NormalRng(std::uint64_t seed) : eng(seed) {}
    double operator()() { return nor(eng); }
};
static_assert(RandomNumberGenerator<NormalRng<>>);

// Pricer system
template <std::invocable<double> Payoff>
class EuropeanPricer
{
    Payoff payoff;
    double df;            // discount factor
    double sum{};
    long   n{};
    double price{};
public:
    EuropeanPricer(Payoff p, double discountFactor)
        : payoff(std::move(p)), df(discountFactor) {}

    void ProcessPath(const std::vector<double>& path)
    {
        sum += payoff(path.back());
        ++n;
    }
    void PostProcess() { price = df * sum / static_cast<double>(n); }
    double Price() const { return price; }
};

// Mediator / SUD
template <StochasticDifferentialEquation Sde,
          FiniteDifferenceScheme<Sde> Fdm,
          RandomNumberGenerator Rng,
          Pricer P>
class MonteCarloEngine
{
    const Sde& sde;   // read-only, shared across engines — race-free
    const Fdm& fdm;
    Rng rng;          // owned: each engine has its own stream
    P   pricer;       // owned: each engine accumulates independently
    int NSim;
public:
    MonteCarloEngine(const Sde& s, const Fdm& f, Rng r, P p, int nSim)
        : sde(s), fdm(f), rng(std::move(r)), pricer(std::move(p)), NSim(nSim) {}

    void run()
    {
        std::vector<double> path(fdm.steps() + 1);
        for (int i = 0; i < NSim; ++i)
        {
            double v = sde.InitialCondition();
            path[0] = v;
            double t = 0.0;
            for (std::size_t nStep = 1; nStep < path.size(); ++nStep)
            {
                v = fdm.advance(v, t, rng());
                path[nStep] = v;
                t += fdm.dt();
            }
            pricer.ProcessPath(path);
        }
        pricer.PostProcess();
    }

    double Price() const { return pricer.Price(); }
};


template <StochasticDifferentialEquation Sde, typename Fdm,
          RandomNumberGenerator Rng, Pricer P>
auto make_engine(const Sde& s, const Fdm& f, Rng r, P p, int nSim)
{
    return MonteCarloEngine<Sde, Fdm, Rng, P>(s, f, std::move(r),
                                              std::move(p), nSim);
}

int main()
{
    constexpr int NSim = 1'000'000;
    constexpr int NT   = 500;

    constexpr double S0  = 60.0, K = 65.0;
    constexpr double vol = 0.3,  r = 0.08, T = 0.25;

    const GBM sde(r, vol, 0.0, S0, T);
    const EulerFdm fdm(sde, NT);             
    const double df = std::exp(-r * T);

    auto callPayoff = [K](double x) { return std::max(0.0, x - K); };
    auto putPayoff  = [K](double x) { return std::max(0.0, K - x); };

    auto e1 = make_engine(sde, fdm, NormalRng<std::mt19937>(1),
                          EuropeanPricer(putPayoff,  df), NSim);
    auto e2 = make_engine(sde, fdm, NormalRng<std::mt19937_64>(2),
                          EuropeanPricer(putPayoff,  df), NSim);
    auto e3 = make_engine(sde, fdm, NormalRng<std::mt19937>(3),
                          EuropeanPricer(callPayoff, df), NSim);
    auto e4 = make_engine(sde, fdm, NormalRng<std::mt19937_64>(4),
                          EuropeanPricer(callPayoff, df), NSim);

    auto t0 = std::chrono::steady_clock::now();
    {
        std::jthread t1([&] { e1.run(); });
        std::jthread t2([&] { e2.run(); });
        std::jthread t3([&] { e3.run(); });
        std::jthread t4([&] { e4.run(); });
    }
    auto el = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count();

    std::cout << "MT32 put:   " << e1.Price() << '\n'
              << "MT64 put:   " << e2.Price() << '\n'
              << "MT32 call:  " << e3.Price() << '\n'
              << "MT64 call:  " << e4.Price() << '\n'
              << "elapsed:    " << el << " s (4 engines in parallel)\n";

    return 0;
}
