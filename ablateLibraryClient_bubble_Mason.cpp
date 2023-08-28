#include <memory>
#include "/Users/jjmarzia/Desktop/ablate/petsc/include/petsc/finclude/petsc.h"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/environment/runEnvironment.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/mathFunctions/functionFactory.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/builder.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/domain/boxMesh.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/domain/modifiers/distributeWithGhostCells.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/domain/modifiers/ghostBoundaryCells.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/eos/perfectGas.hpp"

#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/fluxCalculator/ausm.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/io/interval/fixedInterval.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/monitors/curveMonitor.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/monitors/timeStepMonitor.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/parameters/mapParameters.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/utilities/petscUtilities.hpp"

#include "/Users/jjmarzia/Desktop/ablate/ablate/src/io/hdf5MultiFileSerializer.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/io/interval/simulationTimeInterval.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/compressibleFlowFields.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/fieldFunctions/euler.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/fieldFunctions/compressibleFlowState.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/boundaryConditions/essentialGhost.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/solver/timeStepper.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/mathFunctions/mathFunction.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/finiteVolumeSolver.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/processes/process.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/eos/twoPhase.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/fluxCalculator/riemannStiff.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/processes/surfaceForce.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/domain/initializer.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/finiteVolume/fieldFunctions/densityVolumeFraction.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/mathFunctions/formula.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/mathFunctions/fieldFunction.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/domain/field.hpp"
#include "/Users/jjmarzia/Desktop/ablate/ablate/src/io/interval/fixedInterval.hpp"
#include <string>
#include <iostream>
//hi
//fixed
#include "/Users/jjmarzia/Desktop/ablate/petsc/include/petscerror.h"
#include <vector>
#include <map>



// alot of error are related to {},() becareful of that.... they are different in c++


typedef struct {
    PetscScalar rho;
    PetscScalar vx;
    PetscScalar vy;
    PetscScalar vz;
} InitialConditions;

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscScalar time, const PetscScalar x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho;
    u[ablate::finiteVolume::CompressibleFlowFields::RHOU] = initialConditions->rho * initialConditions-> vx;
    u[ablate::finiteVolume::CompressibleFlowFields::RHOV] = initialConditions->rho * initialConditions-> vy;
    u[ablate::finiteVolume::CompressibleFlowFields::RHOW] = initialConditions->rho * initialConditions-> vz;

    return 0;
}

static PetscErrorCode PhysicsBoundary_Euler(PetscScalar time, const PetscScalar *c, const PetscScalar *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU] = initialConditions->rho * initialConditions-> vx;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOV] = initialConditions->rho * initialConditions-> vy;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOW] = initialConditions->rho * initialConditions-> vz;

    return 0;
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    // initialize petsc and mpi
    ablate::environment::RunEnvironment::Initialize(&argc, &argv);
    ablate::utilities::PetscUtilities::Initialize();

    {

        // define some initial conditions
        InitialConditions initialConditions {
                .rho=1,
                .vx=0,
                .vy=0,
                .vz=0
        };

        // setup the run environment

//        auto runEnvironmentParameters = std::make_shared<ablate::environment::RunEnvironment>
//                )

        ablate::parameters::MapParameters runEnvironmentParameters(
                std::map<std::string, std::string>{
                        {"title", "curvature_AirAir_80x80tt"}
                }
        );

        ablate::environment::RunEnvironment::Setup(runEnvironmentParameters);// runEnvironmentparameter get fixed you need change the path to absolute path in runEnvironmentparameter.hpp


//        auto eos = std::make_shared<ablate::eos::PerfectGas>{
//        ablate::parameters::MapParameters::Create{
//            { "gamma", "0" }, { "Rgas", "0" }
//            }
//        };
//           fixed error but don't know if this is correct way to do it
        auto eos = std::make_shared<ablate::eos::PerfectGas>
                (std::make_shared<ablate::parameters::MapParameters>
                         (std::map<std::string, std::string>{{ "gamma", "0" }, { "Rgas", "0" }})
                );



//        ablate::parameters::MapParameters::Create(
//                {"dm_refine", "1"}
//        )

//        auto eos1 = std::make_shared<ablate::eos::PerfectGas>{ //eosAir
//            ablate::parameters::MapParameters::Create(
//                    { "gamma", "1.4" }, { "Rgas", "287.0" }
//                    )
//        };
        auto eos1 = std::make_shared<ablate::eos::PerfectGas>
                (std::make_shared<ablate::parameters::MapParameters>
                         (std::map<std::string, std::string>{{ "gamma", "1.4" }, { "Rgas", "287.0" }})
                );

        auto* eos2 = &eos1;

//        auto eosTwoPhase = std::make_shared<ablate::eos::TwoPhase>(
//                eos1, eos2
//        );
        auto eosTwoPhase = std::make_shared<ablate::eos::TwoPhase>(
                std::shared_ptr<ablate::eos::PerfectGas> eos1,
        );

        auto fields = std::vector<std::shared_ptr<ablate::domain::FieldDescriptor> >(
            std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos),
            std::make_shared<ablate::domain::FieldDescription>("densityVolumeFraction",
                                                               ablate::domain::FieldType("FVM")),
            std::make_shared<ablate::domain::FieldDescription>("volumeFraction",
                                                               ablate::domain::FieldType("FVM")),
            std::make_shared<ablate::domain::FieldDescription>("pressure",
                                                               ablate::domain::FieldLocation("AUX"),
                                                               ablate::domain::FieldType("FVM"))
        );
//        auto fields = std::vector<std::shared_ptr<ablate::domain::FieldDescriptor> >(
//                std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos),
//                std::make_shared<ablate::domain::FieldDescription>("densityVolumeFraction",
//                                                                   ablate::domain::FieldType("FVM")),
//                std::make_shared<ablate::domain::FieldDescription>("volumeFraction",
//                                                                   ablate::domain::FieldType("FVM")),
//                std::make_shared<ablate::domain::FieldDescription>("pressure",
//                                                                   ablate::domain::FieldLocation("AUX"),
//                                                                   ablate::domain::FieldType("FVM"))
//    };





    auto modifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier> >(
            std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(1),
            std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>("Face Sets")
    );

    auto domain = std::make_shared<ablate::domain::BoxMesh>("simpleBoxField",
                                                            fields,
                                                            modifiers,
                                                            std::vector<int>({40}), //faces
                                                            std::vector<double>({-2, 2}), //upper
                                                            std::vector<double>({-2, 2}), //lower
                                                            std::vector<std::string>{"NONE"} /*boundary*/,
                                                            false /*simplex*/,
                                                            ablate::parameters::MapParameters::Create(
                                                                    {{"dm_refine", "1"},{"dm_distribute", ""}}
                                                            )
    );

    auto io = std::make_shared<ablate::io::Hdf5MultiFileSerializer>(
            std::make_shared<ablate::io::interval::FixedInterval>(0.1)
    );

    // Set up the flow data
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(
            std::map<std::string, std::string>({
                                                       { "cfl", ".5" }}
            )
    );

//         Set the initial conditions for euler
//        auto initialCondition = std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>("euler", ablate::mathFunctions::Create(SetInitialCondition, (void *)&initialConditions));

    auto initialCondition = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", ablate::mathFunctions::Create(SetInitialCondition, (void *)&initialConditions));

    auto initialization = std::vector<std::shared_ptr<ablate::domain::Initializer>>(
            std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(
                    ablate::finiteVolume::fieldFunctions::CompressibleFlowState(
                            eosTwoPhase, //flowfieldState (eos1, eos2),
                            ablate::mathFunctions::ConstantValue(300), //temperature
                            ablate::mathFunctions::Formula("(x^2+y^2) < 1 ? 1e5+0.251036495228486 : ( (x^2+y^2) < 1.1025  ? 1e5+0.209197079357072 : ( (x^2+y^2) < 1.21 ? 1e5+0.167357663485657 : ( (x^2+y^2) < 1.3225 ? 1e5+0.125518247614243 : ( (x^2+y^2) < 1.44 ? 1e5+0.083678831742829 : ( (x^2+y^2) < 1.5625 ? 1e5+0.041839415871414 : 1e5 )))))"), //pressure gradient
                            ablate::mathFunctions::ConstantValue(0), //velocity
                            std::make_shared<ablate::mathFunctions::FieldFunction>("volumeFraction", ablate::mathFunctions::Formula("(x^2+y^2) < 1 ?  0 : ( (x^2+y^2) < 1.1025  ? 0.166666666666667 : ( (x^2+y^2) < 1.21 ? 0.333333333333333 : ( (x^2+y^2) < 1.3225 ? 0.5 : ( (x^2+y^2) < 1.44 ? 0.666666666666667 : ( (x^2+y^2) < 1.5625 ? 0.833333333333333 : 1.0 )))))"))
                    )
            ),
            std::make_shared<ablate::finiteVolume::fieldFunctions::DensityVolumeFraction>(
                    eosTwoPhase
            ),
            std::make_shared<ablate::mathFunctions::FieldFunction>(
                    "volumeFraction",
                    ablate::mathFunctions::Formula("(x^2+y^2) < 1 ?  0 : ( (x^2+y^2) < 1.1025  ? 0.166666666666667 : ( (x^2+y^2) < 1.21 ? 0.333333333333333 : ( (x^2+y^2) < 1.3225 ? 0.5 : ( (x^2+y^2) < 1.44 ? 0.666666666666667 : ( (x^2+y^2) < 1.5625 ? 0.833333333333333 : 1.0 )))))")
            )
    );

    // create a time stepper
    auto timeStepper = ablate::solver::TimeStepper("theMainTimeStepper",
                                                   domain,
            //ablate::parameters::MapParameters::Create({"ts_adapt_type", "physicsConstrained"}, {"ts_max_time", "0.5"}, {"ts_dt", "1e-2"}),
            //
                                                   std::make_shared<ablate::parameters::MapParameters>
                                                           (std::map<std::string, std::vector<std::shared_ptr<ablate::monitors::Monitor>>>({"ts_adapt_type", "physicsConstrained"}, {"ts_max_time", "0.5"}, {"ts_dt", "1e-2"})
                                                           ),
            //
                                                   io,
                                                   initialization);


    auto boundaryConditions = std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>(
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>({1,2,3,4}), ablate::mathFunctions::FieldFunction("euler", "1.1614401858304297, 1.1614401858304297*215250.0, 0.0, 0.0")),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("densityVF", std::vector<int>({1,2,3,4}), ablate::mathFunctions::FieldFunction("densityvolumeFraction", "1.1614401858304297")),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("vf", std::vector<int>({1,2,3,4}), ablate::mathFunctions::FieldFunction("volumeFraction", "1.0"))
    );

    auto processes = std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process> >( //GG, GL, LG, LL
            std::make_shared<ablate::finiteVolume::processes::TwoPhaseEulerAdvection>(
                    eosTwoPhase,
                    parameters,
//                        ablate::finiteVolume::fluxCalculator::RiemannStiff(
//                                eos1,
//                                eos2
//                        ),
//                        ablate::finiteVolume::fluxCalculator::RiemannStiff(
//                                eos1,
//                                eos2
//                        ),
//                        ablate::finiteVolume::fluxCalculator::RiemannStiff(
//                                eos1,
//                                eos2
//                        ),
//                        ablate::finiteVolume::fluxCalculator::RiemannStiff(
//                                eos1,
//                                eos2
//                        )
//                ),
//                std::make_shared<ablate::finiteVolume::processes::SurfaceForce>(
//                        1.125
//                )
                    ablate::finiteVolume::fluxCalculator::RiemannStiff(
                            eos1,
                            eos2
                    ),
                    ablate::finiteVolume::fluxCalculator::RiemannStiff(
                            eos1,
                            eos2
                    ),
                    ablate::finiteVolume::fluxCalculator::RiemannStiff(
                            eos1,
                            eos2
                    ),
                    ablate::finiteVolume::fluxCalculator::RiemannStiff(
                            eos1,
                            eos2
                    )
            ),
            std::make_shared<ablate::finiteVolume::processes::SurfaceForce>(
                    1.125
            )
    );



    auto flowSolver = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>("flow solver", //id, region, options/parameters, processes, boundary conditions
                                                                                 ablate::domain::Region::ENTIREDOMAIN,
                                                                                 processes,
                                                                                 boundaryConditions
    );


    // register the flowSolver with the timeStepper

    timeStepper.Register(
            flowSolver,
            // {std::<ablate::monitors::TimeStepMonitor>(100)}
            std::make_shared<ablate::monitors::TimeStepMonitor>()
    );
    timeStepper.Solve(100);
}

ablate::environment::RunEnvironment::Finalize();
return 0;
}
