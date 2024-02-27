import { useSession, signIn, signOut } from 'next-auth/react'

export default function IndexPage() {
  const { data, status } = useSession();
  if (status === 'authenticated') {
    return (
          <div>
              <h1> hi {data.user.name}</h1>
              <img src={data.user.image} alt={data.user.name + ' photo'} />
          </div>
    );
  } 
  return (
    <div>
      Main page when not signed in
    </div>
  )
}