/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id: metisproto.h,v 1.3 2008-11-17 17:54:33 bscarmo Exp $
 *
 */
/* Undefine the following #define in order to use short int as the idxtype */
#define IDXTYPE_INT

/* Indexes are as long as integers for now */
#ifdef IDXTYPE_INT
typedef int idxtype;
#else
typedef short idxtype;
#endif

void METIS_PartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *,
							  int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphRecursive(int *, idxtype *, idxtype *, idxtype *,
			idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
