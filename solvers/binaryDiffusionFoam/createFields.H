Info<< "Reading field rho\n" << endl;

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field rho_a\n" << endl;

volScalarField rho_a
(
    IOobject
    (
        "rho_a",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Creating field rho_b\n" << endl;

volScalarField rho_b
(
    IOobject
    (
        "rho_b",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    rho - rho_a
);

Info<< "Reading field wa\n" << endl;

volScalarField wa
(
    IOobject
    (
        "wa",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    rho_a/rho
);

Info<< "Reading field wb\n" << endl;

volScalarField wb
(
    IOobject
    (
        "wb",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    rho_b/rho
);

Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

Info<< "Reading diffusion coefficient Dab\n" << endl;

dimensionedScalar Dab("Dab", dimViscosity, transportProperties);

Info<< "Creating field diffusive flux of species A\n" << endl;

volVectorField ja
(
    IOobject
    (
        "ja",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimMass/(dimArea*dimTime), vector::zero)
);

Info<< "Creating field diffusive flux of species B\n" << endl;

volVectorField jb
(
    IOobject
    (
        "jb",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimMass/(dimArea*dimTime), vector::zero)
);

Info<< "Creating field total flux of species A\n" << endl;

volVectorField Na
(
    IOobject
    (
        "Na",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimMass/(dimArea*dimTime), vector::zero)
);

Info<< "Creating field total flux of species B\n" << endl;

volVectorField Nb
(
    IOobject
    (
        "Nb",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimMass/(dimArea*dimTime), vector::zero)
);
Info<< "Creating velocities fields \n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimVelocity, vector::zero)
);

volVectorField U_ver
(
    IOobject
    (
        "U_ver",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("zero", dimVelocity, vector::zero)
);

Info<< "Creating field coeff1\n" << endl;

volScalarField coeff1
(
    IOobject
    (
        "coeff1",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    -Dab*rho/(rho - rho_a)
);
