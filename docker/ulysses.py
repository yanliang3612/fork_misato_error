'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''

#requirements
# git installed
# gcc installed
import os

#fetch source code
if not os.path.exists("ulysses/"):
    os.system("git clone https://gitlab.com/siriius/ulysses.git ulysses")

#build the test file
wfile = open("phoh.xyz","w")
wfile.write('13\n\n')
wfile.write('C          3.23822       -0.14956        0.77394\n')
wfile.write('C          2.25974        0.28499        1.67424\n')
wfile.write('C          0.95814        0.54492        1.22744\n')
wfile.write('C          0.64068        0.36703       -0.12873\n')
wfile.write('O          0.02505        0.96323        2.10620\n')
wfile.write('C          1.62084       -0.06759       -1.02708\n')
wfile.write('C          2.91896       -0.32576       -0.57600\n')
wfile.write('H         -0.88529        1.15364        1.83823\n')
wfile.write('H         -0.36193        0.56461       -0.48655\n')
wfile.write('H          4.24322       -0.34958        1.12244\n')
wfile.write('H          2.51306        0.41957        2.71832\n')
wfile.write('H          1.37440       -0.20418       -2.07212\n')
wfile.write('H          3.67674       -0.66203       -1.27182\n')
wfile.close()

#now build the input program files
#GFN2-xTB calculation
wfile = open("gfn2-xtb.cpp","w")
wfile.write('#include "ulysses/src/GFN.hpp"\n')
wfile.write('#include <stdlib.h>\n\n')
wfile.write('int main(int argc, char** argv) {\n\n')
wfile.write('  char *p;\n')
wfile.write('  int charge = strtol(argv[2],&p,10);\n\n')
wfile.write('  Molecule Mol1(argv[1],charge,1,"C1");\n')
wfile.write('  std::vector<size_t> atoms = Mol1.Atoms();\n')
wfile.write('  matrixE Geometry = Mol1.Geometry();\n\n')
wfile.write('  BSet basis(Mol1,"gfn2");\n')
wfile.write('  std::vector<size_t> AOS = basis.AtomNAOs(atoms);\n\n')
wfile.write('  GFN2 electron(basis,Mol1);\n')
wfile.write('  electron.setSolvent(argv[3]);\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  std::cout << std::setprecision(10);\n')
wfile.write('  std::cout << "energy: " << electron.getEnergy() << " Hartree\\n\\n";\n\n')
wfile.write('  std::vector<double> HOMOorb;\n')
wfile.write('  double Ehomo = electron.getHOMO(HOMOorb);\n')
wfile.write('  std::vector<double> LUMOorb;\n')
wfile.write('  double Elumo = electron.getLUMO(LUMOorb);\n\n')
wfile.write('  std::cout << "HOMO-LUMO gap " << Elumo - Ehomo << std::endl << std::endl;\n\n')
wfile.write('  std::vector<double> AtmCharge = electron.getQAtoms();\n')
wfile.write('  size_t Natoms = AtmCharge.size();\n\n')
wfile.write('  std::vector<double> polarizabilities;\n')
wfile.write('  electron.AtomicPolarizabilities(polarizabilities,AtmCharge);\n')
wfile.write('  std::cout << std::setprecision(5) << "\\n";\n')
wfile.write('  std::cout << "atom       AOs          charge           pol\\n";\n')
wfile.write('  for (size_t idx = 0; idx < Natoms; ++idx) {\n')
wfile.write('    std::cout << atoms[idx] << "          ";\n')
wfile.write('    if (atoms[idx] < 10) {std::cout << " ";}\n')
wfile.write('    std::cout << AOS[idx] << "          ";\n')
wfile.write('    if (AtmCharge[idx] > 0.0) {std::cout << " ";}\n')
wfile.write('    std::cout << AtmCharge[idx] << "          " << polarizabilities[idx] << "\\n";\n')
wfile.write('  }\n')
wfile.write('  std::cout << "\\n";\n\n')
wfile.write('  double polbity = 0.0;\n')
wfile.write('  electron.TotalPolarizability(polbity,AtmCharge);\n')
wfile.write('  std::cout << " Total Polarizability          " << polbity << "\\n";\n\n')
wfile.write('  return 0;\n')
wfile.write('}\n')
wfile.close()

#now compile it
#os.system("g++ gfn2-xtb.cpp -std=c++11 -o gfn2-xtb.exe")
os.system("g++ gfn2-xtb.cpp -std=c++11 -O3 -o gfn2-xtb.exe")

#test run
#input for executable is geometry (this is an old version of the program, therefore only xyz), (total) charge and solvent
os.system("./gfn2-xtb.exe phoh.xyz 0 water")

#AM1 calculation
wfile = open("am1.cpp","w")
wfile.write('#include "ulysses/src/MNDO.hpp"\n')
wfile.write('#include <stdlib.h>\n\n')
wfile.write('int main(int argc, char** argv) {\n\n')
wfile.write('  char *p;\n')
wfile.write('  int charge = strtol(argv[2],&p,10);\n\n')
wfile.write('  Molecule Mol1(argv[1],charge,1,"C1");\n\n')
wfile.write('  std::vector<size_t> atoms = Mol1.Atoms();\n')
wfile.write('  matrixE Geometry = Mol1.Geometry();\n\n')
wfile.write('  BSet basis(Mol1,"am1");\n\n')
wfile.write('  AM1 electron(basis,Mol1);\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  std::cout << std::setprecision(10);\n\n')
wfile.write('  std::cout << "energy: " << electron.getEnergy() << " Hartree\\n\\n";\n\n')
wfile.write('  std::vector<double> HOMOorb;\n')
wfile.write('  double Ehomo = electron.getHOMO(HOMOorb);\n')
wfile.write('  std::vector<double> LUMOorb;\n')
wfile.write('  double Elumo = electron.getLUMO(LUMOorb);\n\n')
wfile.write('  std::cout << "HOMO-LUMO gap " << Elumo - Ehomo << std::endl << std::endl;\n\n')
wfile.write('  std::vector<double> charges = electron.getCharges("Mulliken");\n\n')
wfile.write('  std::cout << "Mulliken charges:" << std::endl;\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  charges = electron.getCharges("CM1");\n\n')
wfile.write('  std::cout << "CM1 charges:" << std::endl;\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  charges = electron.getCharges("CM2");\n\n')
wfile.write('  std::cout << "CM2 charges:" << std::endl;\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  charges = electron.getCharges("CM3");\n\n')
wfile.write('  std::cout << "CM3 charges:" << std::endl;\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  return 0;\n')
wfile.write('}\n')
wfile.close()

#now compile it
#os.system("g++ am1.cpp -std=c++11 -o am1.exe")
os.system("g++ am1.cpp -std=c++11 -O3 -o am1.exe")

#test run
#input for executable is geometry (this is an old version of the program, therefore only xyz), (total) charge
os.system("./am1.exe phoh.xyz 0")

#PM6 calculation
wfile = open("pm6.cpp","w")
wfile.write('#include "ulysses/src/MNDOd.hpp"\n')
wfile.write('#include <stdlib.h>\n\n')
wfile.write('int main(int argc, char** argv) {\n\n')
wfile.write('  char *p;\n')
wfile.write('  int charge = strtol(argv[2],&p,10);\n\n')
wfile.write('  Molecule Mol1(argv[1],charge,1,"C1");\n\n')
wfile.write('  std::vector<size_t> atoms = Mol1.Atoms();\n')
wfile.write('  matrixE Geometry = Mol1.Geometry();\n\n')
wfile.write('  BSet basis(Mol1,"pm6");\n\n')
wfile.write('  PM6 electron(basis,Mol1,"0","D3H4X");\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  std::cout << std::setprecision(10);\n\n')
wfile.write('  std::cout << "energy: " << electron.getEnergy() << " Hartree\\n\\n";\n\n')
wfile.write('  std::vector<double> HOMOorb;\n')
wfile.write('  double Ehomo = electron.getHOMO(HOMOorb);\n')
wfile.write('  std::vector<double> LUMOorb;\n')
wfile.write('  double Elumo = electron.getLUMO(LUMOorb);\n\n')
wfile.write('  std::cout << "HOMO-LUMO gap " << Elumo - Ehomo << std::endl << std::endl;\n\n')
wfile.write('  std::vector<double> charges = electron.getCharges("Mulliken");\n\n')
wfile.write('  std::cout << "Mulliken charges:" << std::endl;\n\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  return 0;\n')
wfile.write('}\n')
wfile.close()

#now compile it
#os.system("g++ pm6.cpp -std=c++11 -o pm6.exe")
os.system("g++ pm6.cpp -std=c++11 -O3 -o pm6.exe")

#test run
#input for executable is geometry (this is an old version of the program, therefore only xyz), (total) charge
os.system("./pm6.exe phoh.xyz 0")

#GFN2-xTB geometry optimization
wfile = open("gfn2-xtb_opt.cpp","w")
wfile.write('#include "ulysses/src/GFN.hpp"\n')
wfile.write('#include "ulysses/src/math/SolverPackage.hpp"\n')
wfile.write('#include <stdlib.h>\n\n')
wfile.write('int main(int argc, char** argv) {\n\n')
wfile.write('  char *p;\n')
wfile.write('  int charge = strtol(argv[2],&p,10);\n\n')
wfile.write('  Molecule Mol1(argv[1],charge,1,"C1");\n\n')
wfile.write('  std::vector<size_t> atoms = Mol1.Atoms();\n')
wfile.write('  size_t Natoms = atoms.size();\n')
wfile.write('  matrixE Geometry = Mol1.Geometry();\n\n')
wfile.write('  BSet basis(Mol1,"gfn2");\n')
wfile.write('  std::vector<size_t> AOS = basis.AtomNAOs(atoms);\n\n')
wfile.write('  GFN2 electron(basis,Mol1);\n')
wfile.write('  electron.setSolvent(argv[3]);\n')
wfile.write('  electron.setRestart(0);\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  BFGSd solve(4,6);\n')
wfile.write('  SolverOpt(electron,solve,4,0,5e-6,1e-3);\n\n')
wfile.write('  std::cout << std::setprecision(10);\n\n')
wfile.write('  Molecule Mol2 = electron.Component();\n')
wfile.write('  Mol2.WriteXYZ(argv[1],1);\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  matrixE optGeometry = electron.Component().Geometry();\n\n')
wfile.write('  std::cout << "optimized energy: " << electron.getEnergy() << " Hartree\\n\\n";\n\n')
wfile.write('  std::vector<double> charges = electron.getCharges("Mulliken");\n\n')
wfile.write('  std::cout << "Mulliken charges:" << std::endl;\n')
wfile.write('  for (size_t idAtm = 0; idAtm < charges.size(); ++idAtm) {\n')
wfile.write('    std::cout << atoms[idAtm] << "    " << charges[idAtm] << std::endl;\n')
wfile.write('  }\n\n')
wfile.write('  return 0;\n')
wfile.write('}\n')
wfile.close()

#now compile it
#os.system("g++ gfn2-xtb_opt.cpp -std=c++11 -o gfn2-xtb_opt.exe")
os.system("g++ gfn2-xtb_opt.cpp -std=c++11 -O3 -o gfn2-xtb_opt.exe")

#test run
#input for executable is geometry (this is an old version of the program, therefore only xyz), (total) charge and solvent
os.system("./gfn2-xtb_opt.exe phoh.xyz 0 water")

#GFN2-xTB geometry optimization
wfile = open("gfn2-xtb_prop.cpp","w")
wfile.write('#include "ulysses/src/GFN.hpp"\n')
wfile.write('#include <stdlib.h>\n\n')
wfile.write('int main(int argc, char** argv) {\n\n')
wfile.write('  char *p;\n')
wfile.write('  int charge = strtol(argv[2],&p,10);\n\n')
wfile.write('  Molecule Mol1(argv[1],charge,1,"C1");\n\n')
wfile.write('  std::vector<size_t> atoms = Mol1.Atoms();\n')
wfile.write('  matrixE Geometry = Mol1.Geometry();\n\n')
wfile.write('  BSet basis(Mol1,"gfn2");\n')
wfile.write('  std::vector<size_t> AOS = basis.AtomNAOs(atoms);\n\n')
wfile.write('  GFN2 electron(basis,Mol1);\n')
wfile.write('  electron.setSolvent(argv[3]);\n\n')
wfile.write('  electron.Calculate(0);\n\n')
wfile.write('  std::cout << std::setprecision(10);\n\n')
wfile.write('  std::cout << "energy: " << electron.getEnergy() << " Hartree\\n\\n";\n\n')
wfile.write('  matrixE RxData(1,1);\n')
wfile.write('  electron.ReactivityIndices(RxData,false);\n')
wfile.write('  std::cout << "Electronic Reactivity indices" << std::endl;\n')
wfile.write('  RxData.Print(4);\n\n')
wfile.write('  electron.ReactivityIndices(RxData,true);\n')
wfile.write('  std::cout << "Orbital Reactivity indices" << std::endl;\n')
wfile.write('  RxData.Print(4);\n\n')
wfile.write('  std::cout << "Ionization Potential (Koopman): " << electron.IonizationPotential(true)*au2eV << "   eV" << std::endl;\n')
wfile.write('  std::cout << "Ionization Potential (Definition): " << electron.IonizationPotential(false)*au2eV << "   eV" << std::endl;\n')
wfile.write('  std::cout << "Electron Affinity (Definition): " << electron.ElectronAffinity()*au2eV << "   eV" << std::endl;\n\n')
wfile.write('  double chi;\n')
wfile.write('  double eta;\n')
wfile.write('  electron.HSABdata(chi,eta);\n')
wfile.write('  std::cout << "Electronegativity: " << chi*au2eV << "   eV" << std::endl;\n')
wfile.write('  std::cout << "Hardness: " << eta*au2eV << "   eV" << std::endl;\n\n')
wfile.write('  return 0;\n')
wfile.write('}\n')
wfile.close()

#now compile it
#os.system("g++ gfn2-xtb_prop.cpp -std=c++11 -o gfn2-xtb_prop.exe")
os.system("g++ gfn2-xtb_prop.cpp -std=c++11 -O3 -o gfn2-xtb_prop.exe")

#test run
#input for executable is geometry (this is an old version of the program, therefore only xyz), (total) charge and solvent
os.system("./gfn2-xtb_prop.exe phoh.xyz 0 water")
