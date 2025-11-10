/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
   equimolarDiffusionFoam

Group
    Meus solvers

Description
    Steady-state solver for binary equimolar contra-diffusion with constant concentration (mass).

    \heading Solver details
    The equation is given by:

    \f[
        \div \left(-Dab \grad rho_a \right)
        = 0
    \f]

    Where:
    \vartable
        rho_a   | Species concentration (mass)
        Da      | Diffusion coefficient
    \endvartable

    \heading Required fields
    \plaintable
        Na      | Total flux [kg/m²s]
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for binary equimolar contra-diffusion with constant concentration (mol)."
    );

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating equimolar diffusion\n" << endl;

    while (simple.loop())
    {
        Info<< "Instante atual = " << runTime.timeName() << nl << endl;

        coeff2 = -Dab*rho/(rho - rho_a*(1 - ratio));

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix rhoiEqn
            (
                fvm::laplacian(coeff2, rho_a)

            );

            rhoiEqn.solve();

        }

        rho_b == rho - rho_a;
        wa == rho_a/rho;
        wb == rho_b/rho;
        ja == -Dab*fvc::grad(rho_a);
        jb == -Dab*fvc::grad(rho_b);

        U ==  ((1 - ratio) / (1.0 + wa*(ratio - 1.0))) * (ja/rho);

        Na == ja + rho_a*U;
        Nb == jb + rho_b*U;

        U_ver == (Na+Nb)/rho;

        runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
