// Code Written by: Hayden Shaddix

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "t1.h"
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLorentzVector.h>

//------------------------------------------------------------------------------
// Particle Class
//
class Particle {
    public:
        Particle();
        Particle(double pt, double eta, double phi, double E); // 4-momentum constructor

        double pt, eta, phi, E, m, p[4];
        void p4(double pT, double eta, double phi, double energy);
        void print() const;
        void setMass(double mass);
        double sintheta() const;
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle() : pt(0.0), eta(0.0), phi(0.0), E(0.0), m(0.0) {
    p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor (4-momentum) --------------------------------------
Particle::Particle(double pt, double eta, double phi, double E) : pt(pt), eta(eta), phi(phi), E(E) {
    // Calculate the 4-momentum components
    p[0] = pt * cos(phi);      // px
    p[1] = pt * sin(phi);      // py
    p[2] = pt * sinh(eta);     // pz
    p[3] = E;                  // energy
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta() const {
    // sintheta = pt / |p| where |p| is the magnitude of the momentum
    double pMag = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    return pt / pMag;
}

void Particle::p4(double pT, double eta, double phi, double energy) {
    pt = pT;
    this->eta = eta;
    this->phi = phi;
    E = energy;
    
    p[0] = pt * cos(phi);   // px
    p[1] = pt * sin(phi);   // py
    p[2] = pt * sinh(eta);  // pz
    p[3] = E;               // energy
}

void Particle::setMass(double mass) {
    m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
void Particle::print() const {
    std::cout << "4-momentum: (" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    std::cout << "Sin(Theta): " << sintheta() << std::endl;
}

//------------------------------------------------------------------------------
// Lepton Class (inherits from Particle)
//------------------------------------------------------------------------------
class Lepton : public Particle {
    public:
        Lepton();
        Lepton(double pt, double eta, double phi, double E, int charge);

        int charge;
        void setCharge(int q);
        void print() const;
};

Lepton::Lepton() : Particle(), charge(0) {}

Lepton::Lepton(double pt, double eta, double phi, double E, int charge) 
    : Particle(pt, eta, phi, E), charge(charge) {}

void Lepton::setCharge(int q) {
    charge = q;
}

void Lepton::print() const {
    Particle::print(); // Call the parent class print function
    std::cout << "Charge: " << charge << std::endl;
}

//------------------------------------------------------------------------------
// Jet Class (inherits from Particle)
//------------------------------------------------------------------------------
class Jet : public Particle {
    public:
        Jet();
        Jet(double pt, double eta, double phi, double E, int hadronFlavour);

        int hadronFlavour;
        void setHadronFlavour(int flavour);
        void print() const;
};

Jet::Jet() : Particle(), hadronFlavour(0) {}

Jet::Jet(double pt, double eta, double phi, double E, int hadronFlavour)
    : Particle(pt, eta, phi, E), hadronFlavour(hadronFlavour) {}

void Jet::setHadronFlavour(int flavour) {
    hadronFlavour = flavour;
}

void Jet::print() const {
    Particle::print(); // Call the parent class print function
    std::cout << "Hadron Flavour: " << hadronFlavour << std::endl;
}

//------------------------------------------------------------------------------
// Main Function
//------------------------------------------------------------------------------
int main() {
    // Input ROOT file and tree
    TFile *f = new TFile("input.root", "READ");
    TTree *t1 = (TTree*)(f->Get("t1"));

    // Declare branches as vectors instead of Float_t
    std::vector<float> *lepPt = nullptr, *lepEta = nullptr, *lepPhi = nullptr, *lepE = nullptr;
    std::vector<int> *lepQ = nullptr;
    std::vector<float> *jetPt = nullptr, *jetEta = nullptr, *jetPhi = nullptr, *jetE = nullptr;
    std::vector<int> *jetHadronFlavour = nullptr;

    // Set branch addresses
    t1->SetBranchAddress("lepPt", &lepPt);
    t1->SetBranchAddress("lepEta", &lepEta);
    t1->SetBranchAddress("lepPhi", &lepPhi);
    t1->SetBranchAddress("lepE", &lepE);
    t1->SetBranchAddress("lepQ", &lepQ);
    
    t1->SetBranchAddress("njets", &njets);
    t1->SetBranchAddress("jetPt", &jetPt);
    t1->SetBranchAddress("jetEta", &jetEta);
    t1->SetBranchAddress("jetPhi", &jetPhi);
    t1->SetBranchAddress("jetE", &jetE);
    t1->SetBranchAddress("jetHadronFlavour", &jetHadronFlavour);

    // Total number of events in ROOT tree
    Long64_t nentries = t1->GetEntries();

    for (Long64_t jentry = 0; jentry < 100; jentry++) {
        t1->GetEntry(jentry);
        std::cout << "Event " << jentry << std::endl;

        // Loop over leptons
        for (size_t i = 0; i < lepPt->size(); ++i) {
            Lepton lepton(lepPt->at(i), lepEta->at(i), lepPhi->at(i), lepE->at(i), lepQ->at(i));
            lepton.print();
        }

        // Loop over jets
        for (size_t i = 0; i < jetPt->size(); ++i) {
            Jet jet(jetPt->at(i), jetEta->at(i), jetPhi->at(i), jetE->at(i), jetHadronFlavour->at(i));
            jet.print();
        }
    }

    return 0;
}

