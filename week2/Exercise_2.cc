// Code Written by: Hayden Shaddix 
// Exercise 2 


#include <iostream>
#include <string>

// Struct to hold information about TAC-HEP students
struct Student {
    std::string name;
    std::string email;
    std::string username;
    std::string experiment;
};

// Function to print the student details without allowing modification
void printStudentInfo(const Student &student) {
    std::cout << "Name: " << student.name << std::endl;
    std::cout << "Email: " << student.email << std::endl;
    std::cout << "Username: " << student.username << std::endl;
    std::cout << "Experiment: " << student.experiment << std::endl;
    std::cout << "-------------------------------" << std::endl;
}

int main() {
    // Creating multiple student objects (Decided to do 3, but would be the same for any number)
    Student student1 = {"Hayden shaddix", "hshaddix@niu.edu", "hshaddix", "ATLAS"};
    Student student2 = {"Fatima Rodriguez", "fatima@niu.edu", "fatima", "Atlas"};
    Student student3 = {"Ashling Quinn", "aq3942@princeton.edu", "aq3942", "CMS"};

    // Printing student information
    printStudentInfo(student1);
    printStudentInfo(student2);
    printStudentInfo(student3);

    return 0;
}
