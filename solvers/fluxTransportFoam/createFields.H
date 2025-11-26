Info<< "Reading field total concentration\n" << endl;

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

Info<< "Creating field wa\n" << endl;

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

Info<< "Creating field wb\n" << endl;

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

Info<< "Reading diffusion coefficient (Dab)\n" << endl;

dimensionedScalar Dab("Dab", dimViscosity, transportProperties);

Info<< "Reading molecular weights (MA and MB)\n" << endl;

dimensionedScalar MA("MA", dimMass/dimMoles, transportProperties);
dimensionedScalar MB("MB", dimMass/dimMoles, transportProperties);

Info<< "Creating field ratio (r)\n" << endl;

dimensionedScalar ratio("ratio", dimless, MB.value()/MA.value());

Info<< "Creating field ja (mass diffusive flux for A) \n" << endl;

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

Info<< "Creating field jb (mass diffusive flux for B) \n" << endl;

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

Info<< "Creating field mass average velocity \n" << endl;

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

Info<< "Creating field Na (total mass flux for A) \n" << endl;

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

Info<< "Creating field Nb (total mass flux for B) \n" << endl;

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

Info<< "Creating field mass average velocity (verification step) \n" << endl;

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
    (Na+Nb)/rho
);

Info<< "Creating field coeff \n" << endl;

volScalarField coeff
(
    IOobject
    (
        "coeff",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    rho/(rho - rho_a)
);
