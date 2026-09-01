#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// Subtype polymorphism, as in the article (S1 style).
class Rng
{
public:
    virtual ~Rng() {}
    virtual double GenerateRn() = 0;
};

class PolarMarsaglia : public Rng
{ 
private:
    std::default_random_engine dre;
    std::uniform_real_distribution<double> unif;
public:
    explicit PolarMarsaglia(unsigned seed = 42)
        : dre(seed), unif(0.0, 1.0) {}

    double GenerateRn() override
    {
        double u, v, S;
        do
        {
            u = 2.0 * unif(dre) - 1.0;
            v = 2.0 * unif(dre) - 1.0;
            S = u * u + v * v;
        } while (S > 1.0 || S <= 0.0);

        double fac = std::sqrt(-2.0 * std::log(S) / S);
        return u * fac;
    }
};

class CPPRng : public Rng
{ 
private:
    std::default_random_engine dre;
    std::normal_distribution<double> nor;
public:
    explicit CPPRng(unsigned seed = 42) : dre(seed), nor(0.0, 1.0) {}
    double GenerateRn() override { return nor(dre); }
};

class MTRng : public Rng
{ 
private:
    std::mt19937 mt;
    std::normal_distribution<double> nor;
public:
    explicit MTRng(unsigned seed = 42) : mt(seed), nor(0.0, 1.0) {}
    double GenerateRn() override { return nor(mt); }
};

class MTRng64 : public Rng
{
private:
    std::mt19937_64 mt;
    std::normal_distribution<double> nor;
public:
    explicit MTRng64(unsigned seed = 42) : mt(seed), nor(0.0, 1.0) {}
    double GenerateRn() override { return nor(mt); }
};

//Pricer system
//Payoff and discounting 
class Pricer
{
public:
    std::function<double(double)> m_payoff;
    std::function<double()>       m_discounter;

    Pricer() = default;
    Pricer(const std::function<double(double)>& payoff,
           const std::function<double()>& discounter)
        : m_payoff(payoff), m_discounter(discounter) {}
    virtual ~Pricer() {}

    virtual void   ProcessPath(const std::vector<double>& arr) = 0;
    virtual void   PostProcess()    = 0;
    virtual double DiscountFactor() = 0;
    virtual double Price()          = 0;
};

class EuropeanPricer : public Pricer
{
private:
    double price;
    double sum;
    long   NSim;
public:
    EuropeanPricer(const std::function<double(double)>& payoff,
                   const std::function<double()>& discounter)
        : Pricer(payoff, discounter), price(0.0), sum(0.0), NSim(0) {}

    void ProcessPath(const std::vector<double>& arr) override
    { 
        // Payoff at terminal time T, accumulated over draws
        sum += m_payoff(arr.back());
        ++NSim;
    }

    double DiscountFactor() override { return m_discounter(); }

    void PostProcess() override
    {
        price = DiscountFactor() * sum / static_cast<double>(NSim); // fix #2
    }

    double Price() override { return price; }
};

//Sde system
class GBM
{ // dX = (mu - d) X dt + vol X dW
private:
    double mu, vol, d, ic, exp_;
public:
    GBM() = default;
    GBM(double driftCoefficient, double diffusionCoefficient,
        double dividendYield, double initialCondition, double expiry)
        : mu(driftCoefficient), vol(diffusionCoefficient),
          d(dividendYield), ic(initialCondition), exp_(expiry) {}

    double Drift(double x, double /*t*/) const { return (mu - d) * x; }
    double Diffusion(double x, double /*t*/) const { return vol * x; }
    double DiffusionDerivative(double /*x*/, double /*t*/) const { return vol; }
    double DriftCorrected(double x, double t, double B) const
    {
        return Drift(x, t) - B * Diffusion(x, t) * DiffusionDerivative(x, t);
    }

    double InitialCondition() const { return ic; }
    double Expiry() const { return exp_; }
};

// Fdm system
template <typename Sde>            
class EulerFdm
{
private:
    std::shared_ptr<Sde> sde;
    int NT;
public:
    std::vector<double> x;  // The mesh array (time discretization points)
    double k;               // Mesh size
    double dtSqrt;

    EulerFdm() = default;
    EulerFdm(const std::shared_ptr<Sde>& stochasticEquation,
             int numSubdivisions)
        : sde(stochasticEquation), NT(numSubdivisions)
    {
        k = sde->Expiry() / static_cast<double>(NT);
        dtSqrt = std::sqrt(k);
        x = std::vector<double>(NT + 1);
        x[0] = 0.0;
        for (std::size_t n = 1; n < x.size(); ++n)
            x[n] = x[n - 1] + k;
    }

    double advance(double xn, double tn, double dt,
                   double normalVar, double /*normalVar2*/) const
    { // Explicit Euler–Maruyama, one step
        return xn + sde->Drift(xn, tn) * dt
                  + sde->Diffusion(xn, tn) * dtSqrt * normalVar;
    }
};

// Mediator / SUD (policy)

template <typename Sde, typename Pricer_, typename Fdm, typename Rng_>
class SUD
{
private:
    std::shared_ptr<Sde>     sde;
    std::shared_ptr<Fdm>     fdm;
    std::shared_ptr<Rng_>    rng;
    std::shared_ptr<Pricer_> pricer;

    int NSim;
    std::vector<double> res;  // Storage for the generated path per simulation
public:
    SUD(const std::shared_ptr<Sde>& s,
        const std::shared_ptr<Pricer_>& p,
        const std::shared_ptr<Fdm>& f,
        const std::shared_ptr<Rng_>& r,
        int numberSimulations, int NT)
        : sde(s), fdm(f), rng(r), pricer(p),
          NSim(numberSimulations), res(NT + 1) {}

    void start()
    { 
        double VOld, VNew;
        for (int i = 1; i <= NSim; ++i)
        {
            VOld = sde->InitialCondition();
            res[0] = VOld;

            for (std::size_t n = 1; n < res.size(); ++n)
            {
                VNew = fdm->advance(VOld, fdm->x[n - 1], fdm->k,
                                    rng->GenerateRn(), rng->GenerateRn());
                res[n] = VNew;
                VOld = VNew;
            }
            pricer->ProcessPath(res);
        }
        pricer->PostProcess();
    }

    double Price() const { return pricer->Price(); }
};

// Timer
class StopWatch
{
    std::chrono::steady_clock::time_point t0;
public:
    void Start() { t0 = std::chrono::steady_clock::now(); }
    double Stop() const
    {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }
};


int main()
{
    const int NSim = 1000000;  
    const int NT   = 500;

    const double driftCoefficient     = 0.08;
    const double diffusionCoefficient = 0.3;
    const double dividendYield        = 0.0;
    const double initialCondition     = 60.0;
    const double expiry               = 0.25;
    const double K                    = 65.0;
    const double r                    = 0.08;
    const double T                    = 0.25;

    auto sde = std::make_shared<GBM>(driftCoefficient, diffusionCoefficient,
                                     dividendYield, initialCondition, expiry);

    std::function<double(double)> payoffPut =
        [K](double x) { return std::max(0.0, K - x); };
    std::function<double(double)> payoffCall =
        [K](double x) { return std::max(0.0, x - K); };
    std::function<double()> discounter =
        [r, T]() { return std::exp(-r * T); };

    auto fdm = std::make_shared<EulerFdm<GBM> >(sde, NT);


    {
        auto pricerPut = std::make_shared<EuropeanPricer>(payoffPut, discounter);
        auto rngCPP    = std::make_shared<CPPRng>(1u);

        StopWatch sw; sw.Start();
        SUD<GBM, EuropeanPricer, EulerFdm<GBM>, CPPRng>
            s(sde, pricerPut, fdm, rngCPP, NSim, NT);
        s.start();
        std::cout << "\n[Sequential] C++11 <random> put: " << s.Price()
                  << "   (" << sw.Stop() << " s)\n";
    }

    auto p1 = std::make_shared<EuropeanPricer>(payoffPut,  discounter);
    auto p2 = std::make_shared<EuropeanPricer>(payoffPut,  discounter);
    auto p3 = std::make_shared<EuropeanPricer>(payoffCall, discounter);
    auto p4 = std::make_shared<EuropeanPricer>(payoffCall, discounter);

    auto rng1 = std::make_shared<PolarMarsaglia>(1u);
    auto rng2 = std::make_shared<MTRng>(2u);
    auto rng3 = std::make_shared<MTRng64>(3u);
    auto rng4 = std::make_shared<CPPRng>(4u);

    SUD<GBM, EuropeanPricer, EulerFdm<GBM>, PolarMarsaglia>
        s1(sde, p1, fdm, rng1, NSim, NT);
    SUD<GBM, EuropeanPricer, EulerFdm<GBM>, MTRng>
        s2(sde, p2, fdm, rng2, NSim, NT);
    SUD<GBM, EuropeanPricer, EulerFdm<GBM>, MTRng64>
        s3(sde, p3, fdm, rng3, NSim, NT);
    SUD<GBM, EuropeanPricer, EulerFdm<GBM>, CPPRng>
        s4(sde, p4, fdm, rng4, NSim, NT);

    auto fn1 = [&]() { s1.start(); };
    auto fn2 = [&]() { s2.start(); };
    auto fn3 = [&]() { s3.start(); };
    auto fn4 = [&]() { s4.start(); };

    {
        StopWatch sw; sw.Start();
        std::thread t1(fn1), t2(fn2), t3(fn3), t4(fn4);
        t1.join(); t2.join(); t3.join(); t4.join();
        double el = sw.Stop();

        std::cout << "\n[std::thread]  Polar Marsaglia put:  " << s1.Price()
                  << "\n               Mersenne 32 put:      " << s2.Price()
                  << "\n               Mersenne 64 call:     " << s3.Price()
                  << "\n               default_engine call:  " << s4.Price()
                  << "\n               elapsed: " << el << " s\n";
    }

    {
        auto q1 = std::make_shared<EuropeanPricer>(payoffPut,  discounter);
        auto q2 = std::make_shared<EuropeanPricer>(payoffCall, discounter);
        auto rA = std::make_shared<MTRng>(11u);
        auto rB = std::make_shared<MTRng>(12u);

        SUD<GBM, EuropeanPricer, EulerFdm<GBM>, MTRng>
            a1(sde, q1, fdm, rA, NSim, NT);
        SUD<GBM, EuropeanPricer, EulerFdm<GBM>, MTRng>
            a2(sde, q2, fdm, rB, NSim, NT);

        StopWatch sw; sw.Start();
        std::future<void> fut1 = std::async(std::launch::async, [&]{ a1.start(); });
        std::future<void> fut2 = std::async(std::launch::async, [&]{ a2.start(); });
        fut1.get();
        fut2.get();
        double el = sw.Stop();

        std::cout << "\n[std::async]   MT put:  " << a1.Price()
                  << "\n               MT call: " << a2.Price()
                  << "\n               elapsed: " << el << " s\n";
    }

    return 0;
}
